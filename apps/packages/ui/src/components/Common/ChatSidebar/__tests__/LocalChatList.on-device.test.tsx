import React from "react";
import userEvent from "@testing-library/user-event";
import { ConfigProvider } from "antd";
import Dexie from "dexie";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { LocalChatList } from "../LocalChatList";
import {
  HistorySelectionProvider,
  useHistorySelectionContext,
  type HistorySelectionController,
} from "@/hooks/chat/useHistorySelection";
import { useStoreMessageOption } from "@/store/option";
import { useStoreChatModelSettings } from "@/store/model";
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror";
import { SETTINGS_NAVIGATION_REQUEST_EVENT } from "@/utils/settings-return";

const io = vi.hoisted(() => ({
  page: vi.fn(),
  preview: vi.fn(),
  metadata: vi.fn(),
  bookmark: vi.fn(),
  owner: vi.fn(),
  capture: vi.fn(),
  ensure: vi.fn(),
  details: vi.fn(),
  model: vi.fn(),
  prompt: vi.fn(),
  files: vi.fn(),
  account: "A",
  resolve: vi.fn(),
  storage: { status: "ready", profileId: "P", epoch: 0 } as {
    status: string;
    profileId: string | null;
    epoch: number;
  },
  navigate: null as ReturnType<typeof useNavigate> | null,
  controller: null as HistorySelectionController | null,
}));
const profileStore = vi.hoisted(() => ({
  settings: undefined as
    | undefined
    | { id: string; history_profile_id?: string; user_id?: string },
  writes: vi.fn(),
}));
const owned = {
  id: "owned",
  title: "Owned local chat",
  local_owner_key: "local-history-v1:P",
  createdAt: 1,
  is_rag: false,
};
vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next");
  const { default: common } = await import("@/assets/locale/en/common.json");
  const instance = createInstance();
  await instance.init({
    lng: "en",
    fallbackLng: "en",
    resources: { en: { common } },
    interpolation: { escapeValue: false },
  });
  return { useTranslation: () => ({ t: instance.t.bind(instance) }) };
});
vi.mock("dexie-react-hooks", () => ({ useLiveQuery: () => io.storage }));
vi.mock("@/db/dexie/schema", () => ({
  db: {
    name: "PageAssistDatabase",
    userSettings: {
      get: async () => profileStore.settings,
      put: async (value: typeof profileStore.settings) => {
        profileStore.writes(value);
        profileStore.settings = value;
      },
    },
    transaction: async (
      _mode: string,
      _tables: unknown[],
      fn: () => Promise<unknown>,
    ) => fn(),
    chatHistories: {},
    messages: {},
  },
}));
vi.mock("@/db/dexie/chat", async (original) => ({
  ...(await original<typeof import("@/db/dexie/chat")>()),
  pageAssistDatabase: {
    getChatHistoriesPaginated: (...args: unknown[]) => io.page(...args),
    getOnDeviceChatPreview: (...args: unknown[]) => io.preview(...args),
    getHistoriesWithMetadata: (...args: unknown[]) => io.metadata(...args),
  },
  PageAssistDatabase: class {
    getHistoryInfo = (...args: unknown[]) => io.details(...args);
    getChatHistory = async () => [];
    getOnDeviceChatPreview = (...args: unknown[]) => io.preview(...args);
  },
}));
vi.mock("@/db/dexie/history-selection", async (original) => ({
  ...(await original<typeof import("@/db/dexie/history-selection")>()),
  ensureLocalProfileId: (...args: unknown[]) => io.ensure(...args),
  loadHistoryBookmark: async () => null,
  saveHistoryBookmark: (...args: unknown[]) => io.bookmark(...args),
  loadHistoryTurnRecoveries: async () => [],
  dismissHistoryTurnRecovery: vi.fn(),
  getLocalHistoryOwner: (...args: unknown[]) => io.owner(...args),
}));
vi.mock("@/db/dexie/helpers", () => ({
  formatToMessage: (rows: unknown[]) => rows,
  formatToChatHistory: (rows: unknown[]) => rows,
  getPromptById: (...args: unknown[]) => io.prompt(...args),
  getSessionFiles: (...args: unknown[]) => io.files(...args),
  generateID: () => "generated",
  getHistoriesWithMetadata: (...args: unknown[]) => io.metadata(...args),
  getFullChatData: async () => ({ historyInfo: owned, messages: [] }),
  deleteByHistoryId: vi.fn(),
  deleteHistoriesByDateRange: vi.fn(),
  updateHistory: vi.fn(),
  pinHistory: vi.fn(),
  restoreChat: vi.fn(),
}));
vi.mock("@/services/service-prompts", () => ({
  isServicePromptScopeUnresolvedError: () => false,
  resolveServicePromptScope: () => io.resolve(),
  subscribeToServicePromptConfigChanges: () => () => {},
  loadServicePromptSnapshot: async () => ({
    requestScope: await io.resolve(),
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: vi.fn(),
  }),
}));
vi.mock("@/services/chat-history-selection", () => ({
  captureHistorySnapshot: (...args: unknown[]) => io.capture(...args),
  confirmLegacyHistoryProjection: vi.fn(),
}));
vi.mock("@/services/model-settings", () => ({
  lastUsedChatModelEnabled: (...args: unknown[]) => io.model(...args),
}));
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => useStoreMessageOption(),
}));
vi.mock("@/hooks/useConnectionState", () => ({ useIsConnected: () => false }));
vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({ showUndoNotification: vi.fn() }),
}));
vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(),
}));
vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => <p>{content}</p>,
}));
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};
function Probe() {
  io.navigate = useNavigate();
  io.controller = useHistorySelectionContext();
  return (
    <output data-testid="route">
      {useLocation().pathname}
      {useLocation().search}
    </output>
  );
}
function mount(
  path = "/chat",
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  props: Partial<React.ComponentProps<typeof LocalChatList>> = {},
  sidebarShell = false,
) {
  const selected = vi.fn();
  const tree = (
    next: Partial<React.ComponentProps<typeof LocalChatList>> = {},
  ) => (
    <ConfigProvider theme={{ token: { motion: false } }}>
      <MemoryRouter initialEntries={[path]}>
        <QueryClientProvider client={client}>
          <HistorySelectionProvider>
            <Probe />
            <div data-chat-sidebar={sidebarShell ? "" : undefined}>
              {sidebarShell && <input aria-label="Find conversations" />}
              <LocalChatList
                searchQuery=""
                selectedChatId={null}
                onSelectChat={selected}
                {...props}
                {...next}
              />
            </div>
          </HistorySelectionProvider>
        </QueryClientProvider>
      </MemoryRouter>
    </ConfigProvider>
  );
  const ui = tree();
  const view = render(ui);
  return {
    ...view,
    client,
    selected,
    ui,
    rerenderList: (next: Partial<React.ComponentProps<typeof LocalChatList>>) =>
      view.rerender(tree(next)),
  };
}
beforeEach(() => {
  vi.clearAllMocks();
  for (const mock of [
    io.page,
    io.preview,
    io.metadata,
    io.bookmark,
    io.owner,
    io.capture,
    io.ensure,
    io.details,
    io.model,
    io.prompt,
    io.files,
    io.resolve,
  ])
    mock.mockReset();
  profileStore.settings = { id: "main", history_profile_id: "P", user_id: "P" };
  io.account = "A";
  io.storage = { status: "ready", profileId: "P", epoch: 0 };
  io.resolve.mockImplementation(async () => ({
    config: { serverUrl: "http://chat.test", authMode: "multi-user" },
    userId: io.account,
  }));
  io.page.mockImplementation(async () => ({
    histories: [owned],
    hasMore: false,
    totalCount: 1,
  }));
  io.preview.mockImplementation(async () => ({
    historyInfo: owned,
    messages: [
      {
        id: "a",
        role: "assistant",
        name: "Cedar",
        content: "Stored reply",
        createdAt: 1,
        images: ["data:image/png;base64,AA=="],
        parent_message_id: "q",
      },
      {
        id: "b",
        role: "assistant",
        name: "Cedar",
        content: "Alternative reply",
        createdAt: 2,
        parent_message_id: "q",
      },
    ],
  }));
  io.prompt.mockResolvedValue(null);
  io.files.mockResolvedValue([]);
  io.ensure.mockResolvedValue("P");
  io.model.mockResolvedValue(false);
  io.details.mockImplementation(async (id: string) => ({ ...owned, id }));
  io.metadata.mockResolvedValue(new Map());
  io.bookmark.mockResolvedValue(undefined);
  io.owner.mockImplementation(async (id: string) => ({
    kind: "local",
    profile_id: "P",
    owner_key: "local-history-v1:P",
    conversation_id: id,
  }));
  io.capture.mockImplementation(async (owner, view) => ({
    status: "captured",
    view: { ...view, owner_key: owner.owner_key },
    snapshot: {
      version: 1,
      owner_key: owner.owner_key,
      conversation_id: owner.conversation_id,
      nodes: [],
      fences: { conversation: "1", history: "1", settings: "1" },
      source_digest: "source",
      storage_context_digest: "context",
      interpretation_status: { kind: "parent_graph_v1" },
    },
    rows: [],
    selected_content: [],
    purpose: "read",
    storage_context_digest: "context",
  }));
  useStoreMessageOption.getState().setTemporaryChat(false);
  useStoreMessageOption.getState().startNewChatDraft(false);
  useStoreMessageOption.setState({
    historyId: null,
    serverChatId: null,
    messages: [],
    history: [],
  });
});
it("does not reuse a warm unqualified history page", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  client.setQueryData(["fetchChatHistory", ""], {
    pages: [
      {
        groups: [
          {
            label: "today",
            items: [{ ...owned, id: "foreign", title: "FOREIGN cached chat" }],
          },
        ],
        hasMore: false,
        totalCount: 1,
      },
    ],
    pageParams: [1],
  });
  mount("/chat", client);
  expect(
    await screen.findByRole("button", { name: "Owned local chat" }),
  ).toBeInTheDocument();
  expect(screen.queryByText("FOREIGN cached chat")).toBeNull();
  expect(io.page.mock.calls.at(-1)?.[2]).toMatchObject({ profileId: "P" });
});
it("distinguishes an initial read failure and retries the same query", async () => {
  io.page.mockRejectedValue(new Error("storage unavailable"));
  mount();
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not load chats on this device",
  );
  io.page.mockResolvedValue({
    histories: [owned],
    hasMore: false,
    totalCount: 1,
  });
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(
    await screen.findByRole("button", { name: "Owned local chat" }),
  ).toBeInTheDocument();
});
it("retains loaded rows and retries a failed next page", async () => {
  io.page.mockImplementation(async (page: number) => {
    if (page === 1) return { histories: [owned], hasMore: true, totalCount: 2 };
    throw new Error("next page unavailable");
  });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: /load more/i }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not load more chats",
  );
  expect(
    screen.getByRole("button", { name: "Owned local chat" }),
  ).toBeInTheDocument();
  io.page.mockImplementation(async (page: number) => ({
    histories:
      page === 1 ? [owned] : [{ ...owned, id: "second", title: "Second chat" }],
    hasMore: page === 1,
    totalCount: 2,
  }));
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(
    await screen.findByRole("button", { name: "Second chat" }),
  ).toBeInTheDocument();
});
it("deduplicates IDs across pages before grouping", async () => {
  io.page.mockImplementation(async (page: number) => ({
    histories: [{ ...owned, is_pinned: page === 2 }],
    hasMore: page === 1,
    totalCount: 2,
  }));
  mount();
  fireEvent.click(await screen.findByRole("button", { name: /load more/i }));
  await waitFor(() =>
    expect(io.page).toHaveBeenCalledWith(
      2,
      undefined,
      expect.anything(),
      expect.objectContaining({ includeCachedServerChats: false }),
    ),
  );
  expect(
    screen.getAllByRole("button", { name: "Owned local chat" }),
  ).toHaveLength(1);
});
it("previews stored alternatives in Temporary without changing real Core or H1 or writing bookmarks", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  useStoreMessageOption.setState({
    messages: [
      { id: "current", isBot: false, message: "Current Temporary turn" },
    ],
    history: [{ role: "user", content: "Current Temporary turn" }],
  });
  const mounted = mount();
  const row = await screen.findByRole("button", {
    name: "Preview chat: Owned local chat",
  });
  const source = useStoreMessageOption.getState();
  const selection = io.controller!.getCurrent();
  const bookmarkCalls = io.bookmark.mock.calls.length;
  fireEvent.click(row);
  expect(await screen.findByRole("dialog")).toHaveTextContent(
    "read-only preview",
  );
  expect(screen.getByRole("dialog")).toHaveTextContent("Stored reply");
  expect(screen.getByRole("dialog")).toHaveTextContent("Alternative reply");
  expect(screen.getByRole("dialog")).toHaveTextContent(
    "Alternative reply 1 of 2",
  );
  expect(screen.getByRole("dialog")).toHaveTextContent(
    "Alternative reply 2 of 2",
  );
  expect(screen.getByRole("dialog").querySelector("img")).not.toBeNull();
  expect(useStoreMessageOption.getState()).toMatchObject({
    temporaryChat: true,
    chatDraftAddress: source.chatDraftAddress,
    chatDraftRevision: source.chatDraftRevision,
    messages: source.messages,
    history: source.history,
  });
  expect(io.controller!.getCurrent()).toEqual(selection);
  expect(io.owner).not.toHaveBeenCalled();
  expect(io.capture).not.toHaveBeenCalled();
  expect(io.bookmark.mock.calls.length).toBe(bookmarkCalls);
  expect(mounted.selected).not.toHaveBeenCalled();
});
it("does not publish a held Temporary preview after New", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  const pending = deferred<unknown>();
  io.preview.mockReturnValueOnce(pending.promise);
  mount();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Preview chat: Owned local chat",
    }),
  );
  await waitFor(() => expect(io.preview).toHaveBeenCalled());
  act(() => useStoreMessageOption.getState().startNewChatDraft());
  await act(async () => {
    pending.resolve({
      historyInfo: owned,
      messages: [{ content: "LATE SECRET" }],
    });
    await pending.promise;
  });
  expect(screen.queryByText("LATE SECRET")).toBeNull();
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
});
it("honors settings-exit cancellation before any owned load", async () => {
  const cancel = (event: Event) => event.preventDefault();
  window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, cancel);
  try {
    const mounted = mount("/settings/chat");
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await act(async () => {
      await Promise.resolve();
    });
    expect(io.owner).not.toHaveBeenCalled();
    expect(io.preview).not.toHaveBeenCalled();
    expect(mounted.selected).not.toHaveBeenCalled();
    expect(screen.getByTestId("route")).toHaveTextContent("/settings/chat");
  } finally {
    window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, cancel);
  }
});
it("navigates an accepted owned selection with the existing qualified handoff", async () => {
  const mounted = mount("/settings/chat");
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("owned"));
  expect(screen.getByTestId("route")).toHaveTextContent(
    "/chat?historySelection=",
  );
  expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({
    kind: "local",
    id: "owned",
    profileId: "P",
  });
});

it("uses real English loading and date labels", async () => {
  const pending = deferred<unknown>();
  io.page.mockReturnValue(pending.promise);
  mount();
  expect(await screen.findByText("Loading…")).toBeInTheDocument();
  await act(async () => {
    pending.resolve({ histories: [owned], totalCount: 1, hasMore: false });
    await pending.promise;
  });
  expect(
    await screen.findByRole("heading", { name: "Older" }),
  ).toBeInTheDocument();
});
it("restores keyboard focus after source invalidation removes a private preview", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  mount();
  const user = userEvent.setup();
  const row = await screen.findByRole("button", {
    name: "Preview chat: Owned local chat",
  });
  await user.click(row);
  await screen.findByText("Stored reply");
  await waitFor(() =>
    expect(document.activeElement?.closest('[role="dialog"]')).not.toBeNull(),
  );
  act(() => useStoreMessageOption.getState().startNewChatDraft());
  await waitFor(() => expect(row).toHaveFocus());
});
it("closes only the preview with Escape and returns focus", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  const mounted = mount();
  const user = userEvent.setup();
  const row = await screen.findByRole("button", {
    name: "Preview chat: Owned local chat",
  });
  await user.click(row);
  await screen.findByText("Stored reply");
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(row).toHaveFocus();
  expect(useStoreMessageOption.getState().temporaryChat).toBe(true);
  expect(mounted.selected).not.toHaveBeenCalled();
});
it("rejects a held preview after a batched profile ABA", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  const pending = deferred<unknown>();
  io.preview.mockReturnValueOnce(pending.promise);
  mount();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Preview chat: Owned local chat",
    }),
  );
  await waitFor(() => expect(io.preview).toHaveBeenCalled());
  act(() => {
    io.storage = { status: "ready", profileId: "P", epoch: 2 };
    Dexie.on.storagemutated.fire({
      "idb://PageAssistDatabase/userSettings/": {} as never,
    });
    Dexie.on.storagemutated.fire({
      "idb://PageAssistDatabase/userSettings/": {} as never,
    });
  });
  await act(async () => {
    pending.resolve({
      historyInfo: owned,
      messages: [{ content: "ABA SECRET" }],
    });
    await pending.promise;
  });
  expect(screen.queryByText("ABA SECRET")).toBeNull();
  expect(screen.queryByRole("dialog")).toBeNull();
});

const realmRow = () => ({
  ...owned,
  local_owner_key: undefined,
  server_scope_key: serverChatMirrorOwnerKey({
    requestScope: {
      config: { serverUrl: "http://chat.test", authMode: "multi-user" },
      userId: "A",
    },
  } as never),
});
function withoutProfile() {
  io.storage = { status: "ready", profileId: null, epoch: 0 };
  const row = realmRow();
  io.details.mockResolvedValue(row);
  io.page.mockResolvedValue({
    histories: [row],
    hasMore: false,
    totalCount: 1,
  });
  io.preview.mockResolvedValue({ historyInfo: row, messages: [] });
}
function profileMutation(profileId: string, epoch: number) {
  io.storage = { status: "ready", profileId, epoch };
  Dexie.on.storagemutated.fire({
    "idb://PageAssistDatabase/userSettings/": {} as never,
  });
}
it("opens a scope-owned chat when initialization finishes before the profile observer", async () => {
  withoutProfile();
  const pending = deferred<string>();
  io.ensure.mockReturnValue(pending.promise);
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(1));
  expect(io.owner).not.toHaveBeenCalled();
  await act(async () => {
    pending.resolve("P");
    await pending.promise;
  });
  expect(mounted.selected).not.toHaveBeenCalled();
  act(() => profileMutation("P", 2));
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledTimes(1));
  expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({
    kind: "local",
    id: "owned",
    profileId: "P",
  });
});
it("opens once when the profile observer resolves before initialization returns", async () => {
  withoutProfile();
  const pending = deferred<string>();
  io.ensure.mockReturnValue(pending.promise);
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(1));
  act(() => profileMutation("P", 1));
  expect(io.owner).not.toHaveBeenCalled();
  await act(async () => {
    pending.resolve("P");
    await pending.promise;
  });
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledTimes(1));
  expect(io.ensure).toHaveBeenCalledTimes(2);
  expect(io.owner).toHaveBeenCalledTimes(1);
  expect(io.capture).toHaveBeenCalledTimes(1);
});
it("does not initialize a local profile for a Temporary scope-owned preview", async () => {
  withoutProfile();
  useStoreMessageOption.getState().setTemporaryChat(true);
  const mounted = mount();
  const source = useStoreMessageOption.getState();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Preview chat: Owned local chat",
    }),
  );
  expect(await screen.findByRole("dialog")).toHaveTextContent(
    "No stored messages",
  );
  expect(io.ensure).not.toHaveBeenCalled();
  expect(io.owner).not.toHaveBeenCalled();
  expect(io.bookmark).not.toHaveBeenCalled();
  expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
    source.chatDraftAddress,
  );
  expect(mounted.selected).not.toHaveBeenCalled();
});
it("does not initialize a profile when settings navigation is rejected", async () => {
  withoutProfile();
  const cancel = (event: Event) => event.preventDefault();
  window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, cancel);
  try {
    mount("/settings/chat");
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await act(async () => {
      await Promise.resolve();
    });
    expect(io.ensure).not.toHaveBeenCalled();
    expect(io.owner).not.toHaveBeenCalled();
  } finally {
    window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, cancel);
  }
});
it.each(["New", "account ABA", "unmount"])(
  "cancels held profile preparation after %s",
  async (reason) => {
    withoutProfile();
    const pending = deferred<string>();
    io.ensure.mockReturnValue(pending.promise);
    const mounted = mount();
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(1));
    if (reason === "New")
      act(() => useStoreMessageOption.getState().startNewChatDraft());
    if (reason === "account ABA") {
      act(() => {
        io.account = "B";
        window.dispatchEvent(new Event("tldw:auth-principal-changed"));
        io.account = "A";
        window.dispatchEvent(new Event("tldw:auth-principal-changed"));
      });
      await act(async () => {
        await Promise.resolve();
      });
    }
    if (reason === "unmount") mounted.unmount();
    const source = useStoreMessageOption.getState();
    await act(async () => {
      pending.resolve("P");
      await pending.promise;
      profileMutation("P", 2);
    });
    expect(io.owner).not.toHaveBeenCalled();
    expect(io.capture).not.toHaveBeenCalled();
    expect(mounted.selected).not.toHaveBeenCalled();
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
      source.chatDraftAddress,
    );
  },
);
it("refuses a different observed profile after initialization returns", async () => {
  withoutProfile();
  const pending = deferred<string>();
  io.ensure.mockReturnValue(pending.promise);
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(1));
  await act(async () => {
    pending.resolve("P");
    await pending.promise;
  });
  act(() => profileMutation("Q", 2));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not open this chat",
  );
  expect(io.owner).not.toHaveBeenCalled();
  expect(io.capture).not.toHaveBeenCalled();
  expect(mounted.selected).not.toHaveBeenCalled();
});
it("does not steal focus from a newly activated control on preview invalidation", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  mount();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Preview chat: Owned local chat",
    }),
  );
  await screen.findByText("Stored reply");
  const outside = document.createElement("button");
  outside.textContent = "New outside control";
  document.body.append(outside);
  try {
    act(() => {
      outside.focus();
      useStoreMessageOption.getState().startNewChatDraft();
    });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(outside).toHaveFocus();
  } finally {
    outside.remove();
  }
});
it("labels comparison prompts and results separately from alternative replies", async () => {
  useStoreMessageOption.getState().setTemporaryChat(true);
  io.preview.mockResolvedValue({
    historyInfo: owned,
    messages: [
      {
        id: "q",
        role: "user",
        content: "Compare prompt",
        messageType: "compare:user",
        clusterId: "c",
      },
      {
        id: "a",
        role: "assistant",
        content: "Compare A",
        messageType: "compare:assistant",
        clusterId: "c",
        parent_message_id: "q",
      },
      {
        id: "b",
        role: "assistant",
        content: "Compare B",
        messageType: "compare:assistant",
        clusterId: "c",
        parent_message_id: "q",
      },
    ],
  });
  mount();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Preview chat: Owned local chat",
    }),
  );
  expect(await screen.findByText("Comparison prompt")).toBeInTheDocument();
  expect(screen.getAllByText("Comparison result")).toHaveLength(2);
  expect(screen.queryByText("Alternative reply 1 of 2")).toBeNull();
});

it.each(["failed", "unmounted", "New"])(
  "does not report or publish an unaccepted normal selection after %s",
  async (outcome) => {
    const pending = deferred<unknown>();
    io.preview.mockReturnValueOnce(pending.promise);
    const mounted = mount();
    const source = useStoreMessageOption.getState();
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await waitFor(() => expect(io.preview).toHaveBeenCalled());
    if (outcome === "unmounted") mounted.unmount();
    if (outcome === "New")
      act(() => useStoreMessageOption.getState().startNewChatDraft());
    const current = useStoreMessageOption.getState();
    if (outcome === "failed")
      vi.spyOn(console, "error").mockImplementation(() => undefined);
    await act(async () => {
      if (outcome === "failed")
        pending.reject(new Error("Synthetic read failure"));
      else pending.resolve({ historyInfo: owned, messages: [] });
      await pending.promise.catch(() => undefined);
    });
    expect(mounted.selected).not.toHaveBeenCalled();
    expect(io.owner).not.toHaveBeenCalled();
    expect(io.capture).not.toHaveBeenCalled();
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
      outcome === "New" ? current.chatDraftAddress : source.chatDraftAddress,
    );
    if (outcome === "failed")
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Could not open this chat",
      );
  },
);
it("keeps a settled scope-owned H1 lease when only the on-device list closes", async () => {
  const row = realmRow();
  io.details.mockResolvedValue(row);
  io.page.mockResolvedValue({
    histories: [row],
    hasMore: false,
    totalCount: 1,
  });
  io.preview.mockResolvedValue({ historyInfo: row, messages: [] });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const selected = vi.fn();
  const content = (visible: boolean) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <HistorySelectionProvider>
          <Probe />
          {visible && <LocalChatList searchQuery="" onSelectChat={selected} />}
        </HistorySelectionProvider>
      </QueryClientProvider>
    </MemoryRouter>
  );
  const page = render(content(true));
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(selected).toHaveBeenCalledOnce());
  const owner = io.controller!.getCurrent().owner!;
  expect(owner.validate_lease?.()).toBe(true);
  page.rerender(content(false));
  expect(io.controller!.getCurrent().owner).toBe(owner);
  expect(owner.validate_lease?.()).toBe(true);
  act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")));
  expect(owner.validate_lease?.()).toBe(false);
});
it.each(["preparation", "qualified read"])(
  "cancels a held %s when the initiating settings route changes",
  async (boundary) => {
    const pending = deferred<unknown>();
    if (boundary === "preparation") {
      withoutProfile();
      io.ensure.mockReturnValue(pending.promise);
    } else io.preview.mockReturnValueOnce(pending.promise);
    const mounted = mount("/settings/chat");
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await waitFor(() =>
      expect(
        boundary === "preparation" ? io.ensure : io.preview,
      ).toHaveBeenCalled(),
    );
    act(() => io.navigate!("/settings/other"));
    await act(async () => {
      pending.resolve(
        boundary === "preparation" ? "P" : { historyInfo: owned, messages: [] },
      );
      await pending.promise;
      if (boundary === "preparation") profileMutation("P", 2);
    });
    expect(screen.getByTestId("route")).toHaveTextContent("/settings/other");
    expect(mounted.selected).not.toHaveBeenCalled();
    expect(io.owner).not.toHaveBeenCalled();
    expect(io.capture).not.toHaveBeenCalled();
  },
);

it("rejects promotion during preparation before replacing a settled H1 owner", async () => {
  const first = realmRow();
  let second = { ...realmRow(), id: "second", title: "Second local chat" };
  io.page.mockResolvedValue({
    histories: [first, second],
    hasMore: false,
    totalCount: 2,
  });
  io.preview.mockImplementation(async (id: string) => ({
    historyInfo: id === first.id ? first : second,
    messages: [],
  }));
  io.details.mockImplementation(async (id: string) =>
    id === first.id ? first : second,
  );
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("owned"));
  const state = useStoreMessageOption.getState();
  const selection = io.controller!.getCurrent();
  const captures = io.capture.mock.calls.length;
  expect(selection.owner?.validate_lease?.()).toBe(true);
  const pending = deferred<boolean>();
  io.model.mockReturnValueOnce(pending.promise);
  fireEvent.click(screen.getByRole("button", { name: "Second local chat" }));
  await waitFor(() => expect(io.model).toHaveBeenCalledTimes(2));
  second = { ...second, server_chat_id: "native-second" } as typeof second;
  await act(async () => {
    pending.resolve(false);
    await pending.promise;
  });
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not open this chat",
  );
  expect(io.capture.mock.calls.length).toBe(captures);
  expect(mounted.selected).toHaveBeenCalledTimes(1);
  expect(io.controller!.getCurrent().owner).toBe(selection.owner);
  expect(selection.owner?.validate_lease?.()).toBe(true);
  expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
    state.chatDraftAddress,
  );
  expect(useStoreMessageOption.getState().messages).toBe(state.messages);
});

it.each(["model", "prompt", "files"] as const)(
  "preserves a settled Core/H1 source when %s preparation fails",
  async (boundary) => {
    const first = realmRow();
    const second = {
      ...realmRow(),
      id: "second",
      title: "Second local chat",
      last_used_prompt: { prompt_id: "prompt", prompt_content: "" },
    };
    io.page.mockResolvedValue({
      histories: [first, second],
      hasMore: false,
      totalCount: 2,
    });
    io.preview.mockImplementation(async (id: string) => ({
      historyInfo: id === first.id ? first : second,
      messages: [],
    }));
    io.details.mockImplementation(async (id: string) =>
      id === first.id ? first : second,
    );
    const mounted = mount();
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("owned"));
    const state = useStoreMessageOption.getState();
    const selection = io.controller!.getCurrent();
    const captures = io.capture.mock.calls.length;
    const sourceTitle = document.title;
    const read = { model: io.model, prompt: io.prompt, files: io.files }[
      boundary
    ];
    read.mockRejectedValueOnce(new Error("Synthetic preparation failure"));
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    fireEvent.click(screen.getByRole("button", { name: "Second local chat" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not open this chat",
    );
    expect(io.capture.mock.calls.length).toBe(captures);
    expect(mounted.selected).toHaveBeenCalledTimes(1);
    expect(io.controller!.getCurrent().owner).toBe(selection.owner);
    expect(io.controller!.getCurrent().view).toBe(selection.view);
    expect(selection.owner?.validate_lease?.()).toBe(true);
    expect(useStoreMessageOption.getState()).toMatchObject({
      chatDraftAddress: state.chatDraftAddress,
      chatDraftRevision: state.chatDraftRevision,
      messages: state.messages,
      history: state.history,
      historyId: state.historyId,
      serverChatId: state.serverChatId,
      contextFiles: state.contextFiles,
      selectedModel: state.selectedModel,
      selectedSystemPrompt: state.selectedSystemPrompt,
    });
    expect(document.title).toBe(sourceTitle);
  },
);
it("rejects prepared profile ABA even when the final profile matches", async () => {
  withoutProfile();
  const pending = deferred<string>();
  io.ensure.mockReturnValueOnce(pending.promise);
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(1));
  act(() => {
    profileMutation("Q", 1);
    profileMutation("P", 2);
  });
  await act(async () => {
    pending.resolve("P");
    await pending.promise;
  });
  expect(io.owner).not.toHaveBeenCalled();
  expect(io.capture).not.toHaveBeenCalled();
  expect(mounted.selected).not.toHaveBeenCalled();
});
it("accepts only the latest null-profile row click", async () => {
  withoutProfile();
  const first = deferred<string>();
  const second = deferred<string>();
  io.ensure
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise);
  const rows = [
    realmRow(),
    { ...realmRow(), id: "second", title: "Second local chat" },
  ];
  io.page.mockResolvedValue({ histories: rows, hasMore: false, totalCount: 2 });
  io.preview.mockImplementation(async (id: string) => ({
    historyInfo: rows.find((row) => row.id === id),
    messages: [],
  }));
  io.details.mockImplementation(async (id: string) =>
    rows.find((row) => row.id === id),
  );
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Second local chat" }));
  await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(2));
  await act(async () => {
    first.resolve("P");
    await first.promise;
  });
  expect(mounted.selected).not.toHaveBeenCalled();
  await act(async () => {
    second.resolve("P");
    await second.promise;
  });
  act(() => profileMutation("P", 2));
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("second"));
  expect(mounted.selected).toHaveBeenCalledTimes(1);
  expect(io.owner).toHaveBeenCalledWith(
    "second",
    expect.objectContaining({ profileId: "P" }),
  );
});
it("does not reopen after a settings route ABA during preparation", async () => {
  withoutProfile();
  const pending = deferred<string>();
  io.ensure.mockReturnValueOnce(pending.promise);
  const mounted = mount("/settings/chat");
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(io.ensure).toHaveBeenCalledTimes(1));
  act(() => io.navigate!("/settings/other"));
  await act(async () => {
    await Promise.resolve();
  });
  act(() => io.navigate!(-1));
  await act(async () => {
    await Promise.resolve();
  });
  await act(async () => {
    pending.resolve("P");
    await pending.promise;
    profileMutation("P", 2);
  });
  expect(screen.getByTestId("route")).toHaveTextContent("/settings/chat");
  expect(mounted.selected).not.toHaveBeenCalled();
  expect(io.owner).not.toHaveBeenCalled();
  expect(io.capture).not.toHaveBeenCalled();
});
it.each(["New", "account ABA"])(
  "revokes a settled owner after list closure and %s",
  async (boundary) => {
    const row = realmRow();
    io.page.mockResolvedValue({
      histories: [row],
      hasMore: false,
      totalCount: 1,
    });
    io.preview.mockResolvedValue({ historyInfo: row, messages: [] });
    io.details.mockResolvedValue(row);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const selected = vi.fn();
    const content = (visible: boolean) => (
      <MemoryRouter>
        <QueryClientProvider client={client}>
          <HistorySelectionProvider>
            <Probe />
            {visible && (
              <LocalChatList searchQuery="" onSelectChat={selected} />
            )}
          </HistorySelectionProvider>
        </QueryClientProvider>
      </MemoryRouter>
    );
    const page = render(content(true));
    fireEvent.click(
      await screen.findByRole("button", { name: "Owned local chat" }),
    );
    await waitFor(() => expect(selected).toHaveBeenCalledOnce());
    const owner = io.controller!.getCurrent().owner!;
    page.rerender(content(false));
    expect(owner.validate_lease?.()).toBe(true);
    if (boundary === "New")
      act(() => useStoreMessageOption.getState().startNewChatDraft());
    else
      act(() => {
        io.account = "B";
        window.dispatchEvent(new Event("tldw:auth-principal-changed"));
        io.account = "A";
        window.dispatchEvent(new Event("tldw:auth-principal-changed"));
      });
    expect(owner.validate_lease?.()).toBe(false);
  },
);

it("preserves the previous owned selection when candidate qualification fails", async () => {
  const first = realmRow();
  const second = { ...realmRow(), id: "second", title: "Second local chat" };
  io.page.mockResolvedValue({
    histories: [first, second],
    hasMore: false,
    totalCount: 2,
  });
  io.preview.mockImplementation(async (id: string) => ({
    historyInfo: id === first.id ? first : second,
    messages: [],
  }));
  io.details.mockImplementation(async (id: string) =>
    id === first.id ? first : second,
  );
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("owned"));
  const state = useStoreMessageOption.getState();
  const selection = io.controller!.getCurrent();
  const captures = io.capture.mock.calls.length;
  const bookmarks = io.bookmark.mock.calls.length;
  const pending = deferred<unknown>();
  io.owner.mockReturnValueOnce(pending.promise);
  fireEvent.click(screen.getByRole("button", { name: "Second local chat" }));
  await waitFor(() => expect(io.owner).toHaveBeenCalledTimes(2));
  await act(async () => {
    pending.reject(new Error("owner_mismatch"));
    await pending.promise.catch(() => undefined);
  });
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not open this chat",
  );
  expect(io.capture.mock.calls.length).toBe(captures);
  expect(io.bookmark.mock.calls.length).toBe(bookmarks);
  expect(mounted.selected).toHaveBeenCalledTimes(1);
  expect(io.controller!.getCurrent().owner).toBe(selection.owner);
  expect(io.controller!.getCurrent().view).toBe(selection.view);
  expect(selection.owner?.validate_lease?.()).toBe(true);
  expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
    state.chatDraftAddress,
  );
});

it("does not initialize a replacement profile if it disappears before H1 open", async () => {
  const { ensureLocalProfileId: actualEnsure } = await vi.importActual<
    typeof import("@/db/dexie/history-selection")
  >("@/db/dexie/history-selection");
  io.ensure.mockImplementation(actualEnsure);
  const first = realmRow();
  const second = { ...realmRow(), id: "second", title: "Second local chat" };
  io.page.mockResolvedValue({
    histories: [first, second],
    hasMore: false,
    totalCount: 2,
  });
  io.preview.mockImplementation(async (id: string) => ({
    historyInfo: id === first.id ? first : second,
    messages: [],
  }));
  io.details.mockImplementation(async (id: string) =>
    id === first.id ? first : second,
  );
  const mounted = mount();
  fireEvent.click(
    await screen.findByRole("button", { name: "Owned local chat" }),
  );
  await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("owned"));
  const source = useStoreMessageOption.getState();
  const captures = io.capture.mock.calls.length;
  const bookmarks = io.bookmark.mock.calls.length;
  const settled = io.controller!.getCurrent();
  io.owner.mockImplementationOnce(async (id: string) => {
    // The DB changed before its UI observer; the real profile helper must fail read-only.
    profileStore.settings = undefined;
    return {
      kind: "local",
      profile_id: "P",
      owner_key: "local-history-v1:P",
      conversation_id: id,
    };
  });
  fireEvent.click(screen.getByRole("button", { name: "Second local chat" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not open this chat",
  );
  expect(profileStore.writes).not.toHaveBeenCalled();
  expect(profileStore.settings).toBeUndefined();
  expect(io.capture.mock.calls.length).toBe(captures);
  expect(io.bookmark.mock.calls.length).toBe(bookmarks);
  expect(mounted.selected).toHaveBeenCalledTimes(1);
  expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
    source.chatDraftAddress,
  );
  expect(io.controller!.getCurrent().owner).toBe(settled.owner);
  expect(io.controller!.getCurrent().view).toBe(settled.view);
});

it("previews an owned cached Server copy in normal mode without changing the source draft or adopting H1", async () => {
  const ownerKey = serverChatMirrorOwnerKey({
    requestScope: await io.resolve(),
  });
  const cached = {
    ...owned,
    id: "copy",
    title: "Cached Cedar",
    server_chat_id: "S",
    server_scope_key: ownerKey,
    server_scope_type: "global",
  };
  io.page.mockResolvedValue({
    histories: [cached],
    totalCount: 1,
    hasMore: false,
  });
  io.preview.mockResolvedValue({
    historyInfo: cached,
    messages: [
      {
        id: "stored",
        role: "assistant",
        name: "Cedar",
        content: "Cached opening",
        createdAt: 1,
      },
    ],
  });
  const before = useStoreMessageOption.getState();
  const view = mount("/chat", undefined, { includeCachedServerChats: true });
  const trigger = await screen.findByRole("button", {
    name: "Preview cached server chat: Cached Cedar · copy",
  });
  fireEvent.click(trigger);
  expect(await screen.findByText("Cached opening")).toBeInTheDocument();
  await waitFor(() =>
    expect(
      screen.getByText(/The copy may be incomplete or contain local edits/),
    ).toBeVisible(),
  );
  expect(view.selected).not.toHaveBeenCalled();
  expect(io.capture).not.toHaveBeenCalled();
  expect(io.bookmark).not.toHaveBeenCalled();
  expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(
    before.chatDraftAddress,
  );
  expect(useStoreMessageOption.getState().messages).toEqual(before.messages);
  fireEvent.click(screen.getByRole("button", { name: "Close preview" }));
  await waitFor(() => expect(trigger).toHaveFocus());
});

it("restores the stored copy when the exact visible Server receipt is retired", async () => {
  const ownerKey = serverChatMirrorOwnerKey({
    requestScope: await io.resolve(),
  });
  const cached = {
    ...owned,
    id: "copy",
    title: "Cached Cedar",
    server_chat_id: "S",
    server_scope_key: ownerKey,
    server_scope_type: "global",
  };
  const { isReadableOnDeviceHistory } =
    await import("@/db/dexie/history-selection");
  io.page.mockImplementation(async (_page, _query, scope, options) => {
    const histories = [owned, cached].filter((row) =>
      isReadableOnDeviceHistory(row, scope, options),
    );
    return { histories, totalCount: histories.length, hasMore: false };
  });
  let current = true;
  const receipt = {
    serverScopeKey: ownerKey,
    query: "",
    ids: new Set(["S"]),
    isCurrent: () => current,
  };
  const view = mount("/chat", undefined, {
    includeCachedServerChats: true,
    visibleServerChats: receipt,
  });
  await screen.findByRole("button", { name: "Owned local chat" });
  expect(
    screen.queryByRole("button", {
      name: "Preview cached server chat: Cached Cedar · copy",
    }),
  ).toBeNull();
  expect(io.page.mock.results.some((result) => result.type === "return")).toBe(
    true,
  );
  const successfulPage = await io.page.mock.results.at(-1)!.value;
  expect(successfulPage.histories.map((row: typeof owned) => row.id)).toEqual([
    "owned",
  ]);
  const readsBefore = io.page.mock.calls.length;
  current = false;
  view.rerenderList({});
  expect(
    await screen.findByRole("button", {
      name: "Preview cached server chat: Cached Cedar · copy",
    }),
  ).toBeInTheDocument();
  expect(io.page.mock.calls.length).toBeGreaterThan(readsBefore);
});

it("explains failed cached-copy account verification while leaving profile-owned chats readable and retryable", async () => {
  io.resolve.mockRejectedValue(new Error("account verification unavailable"));
  mount("/chat", undefined, { includeCachedServerChats: true });
  expect(
    await screen.findByRole("button", { name: "Owned local chat" }),
  ).toBeInTheDocument();
  expect(
    await screen.findByText(
      "Could not verify the account for cached Server copies. Profile-owned chats are still available.",
    ),
  ).toBeVisible();
  const attempts = io.resolve.mock.calls.length;
  fireEvent.click(
    screen.getByRole("button", { name: "Retry account verification" }),
  );
  await waitFor(() =>
    expect(io.resolve.mock.calls.length).toBeGreaterThan(attempts),
  );
});

it("distinguishes equal local titles with visible and accessible stored IDs", async () => {
  io.page.mockResolvedValue({
    histories: [
      { ...owned, id: "A", title: "Same title" },
      { ...owned, id: "B", title: "Same title" },
    ],
    totalCount: 2,
    hasMore: false,
  });
  mount();
  expect(
    await screen.findByRole("button", { name: "Same title · Chat A" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Same title · Chat B" }),
  ).toBeInTheDocument();
  expect(screen.getByText("Chat A")).toBeVisible();
});

it("returns to Find after matching Server results remove the cached preview trigger", async () => {
  const user = userEvent.setup();
  const ownerKey = serverChatMirrorOwnerKey({
    requestScope: await io.resolve(),
  });
  const cached = {
    ...owned,
    id: "copy",
    title: "Cached Cedar",
    server_chat_id: "S",
    server_scope_key: ownerKey,
    server_scope_type: "global",
  };
  io.page.mockResolvedValue({
    histories: [cached],
    totalCount: 1,
    hasMore: false,
  });
  io.preview.mockResolvedValue({ historyInfo: cached, messages: [] });
  const view = mount(
    "/chat",
    undefined,
    { includeCachedServerChats: true },
    true,
  );
  const trigger = await screen.findByRole("button", {
    name: "Preview cached server chat: Cached Cedar · copy",
  });
  await user.click(trigger);
  await screen.findByRole("dialog");
  view.rerenderList({
    visibleServerChats: {
      serverScopeKey: ownerKey,
      query: "",
      ids: new Set(["S"]),
      isCurrent: () => true,
    },
  });
  expect(trigger).not.toBeInTheDocument();
  await user.keyboard("{Escape}");
  await waitFor(() =>
    expect(
      screen.getByRole("textbox", { name: "Find conversations" }),
    ).toHaveFocus(),
  );
});

// These component cases use the same profile-qualified reader as the real sidebar.
it.each(["current", "unmounted", "failed"] as const)(
  "reports only an accepted LocalChatList selection: %s",
  async outcome => {
    const row = { ...owned, last_used_prompt: { prompt_id: "saved-prompt", prompt_content: "" } };
    io.details.mockResolvedValue(row);
    io.prompt.mockResolvedValue({ id: "saved-prompt", content: "Saved prompt" });
    const pending = deferred<unknown>();
    io.preview.mockReturnValueOnce(pending.promise);
    useStoreMessageOption.setState({ historyId: "before" });
    useStoreChatModelSettings.getState().setSystemPrompt("before");
    const mounted = mount();
    fireEvent.click(await screen.findByRole("button", { name: "Owned local chat" }));
    await waitFor(() => expect(io.preview).toHaveBeenCalled());
    if (outcome === "unmounted") mounted.unmount();
    if (outcome === "failed") vi.spyOn(console, "error").mockImplementation(() => undefined);
    await act(async () => {
      if (outcome === "failed") pending.reject(new Error("Synthetic read failure"));
      else pending.resolve({ historyInfo: row, messages: [{ content: "Current local chat" }] });
      await pending.promise.catch(() => undefined);
    });
    if (outcome === "current") {
      await waitFor(() => expect(mounted.selected).toHaveBeenCalledWith("owned"));
      expect(useStoreMessageOption.getState().historyId).toBe("owned");
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe("Saved prompt");
    } else {
      expect(mounted.selected).not.toHaveBeenCalled();
      expect(useStoreMessageOption.getState().historyId).toBe("before");
      if (outcome === "failed") expect(await screen.findByRole("alert")).toHaveTextContent("Could not open this chat");
    }
    mounted.unmount();
    mounted.client.clear();
  },
);
