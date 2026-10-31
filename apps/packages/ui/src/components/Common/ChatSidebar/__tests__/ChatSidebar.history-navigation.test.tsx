// @vitest-environment jsdom
import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { MemoryRouter, useNavigate } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import i18next from "i18next";
import ICU from "@/i18n/icu-format";
import { I18nextProvider } from "react-i18next";
import common from "@/assets/locale/en/common.json";
import { ChatSidebar } from "../../ChatSidebar";
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator";
import * as settings from "@/services/settings/ui-settings";
import { getSetting, __clearSettingCachesForTests } from "@/services/settings/registry";

type Change = { newValue: unknown; oldValue: unknown };
const io = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  listeners: new Map<string, Set<(change: Change) => void>>(),
  reads: new Map<string, () => Promise<unknown>>(),
  writes: vi.fn(),
  history: vi.fn(),
  historyTotal: 0,
  temporary: false,
  ownerCurrent: true,
  verificationState: "ready",
  requestScope: {config: {serverUrl: "http://chat.test", authMode: "multi-user" as const}, userId: "A"},
  folderData: {} as Record<string, unknown>,
  folderProps: vi.fn(),
  verify: vi.fn(),
}));
// Real useSetting, registry and Plasmo hook; the storage IO boundary is in memory.
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) =>
      io.reads.has(key) ? io.reads.get(key)!() : io.values.get(key),
    set: async (key: string, value: unknown) => {
      const oldValue = io.values.get(key);
      io.writes(key, value);
      io.values.set(key, value);
      io.listeners
        .get(key)
        ?.forEach((listener) => listener({ newValue: value, oldValue }));
    },
    watch: (listeners: Record<string, (change: Change) => void>) => {
      for (const [key, listener] of Object.entries(listeners)) {
        const callbacks = io.listeners.get(key) ?? new Set();
        callbacks.add(listener);
        io.listeners.set(key, callbacks);
      }
      return () => {
        for (const [key, listener] of Object.entries(listeners))
          io.listeners.get(key)?.delete(listener);
      };
    },
    unwatch: (listeners: Record<string, (change: Change) => void>) => {
      for (const [key, listener] of Object.entries(listeners))
        io.listeners.get(key)?.delete(listener);
    },
  }),
}));
vi.mock("@/hooks/useChatDraftOwner", () => ({
  useChatDraftOwner: () => ({
    ownerKey: "A",
    serverScopeKey: "A",
    requestScope: io.requestScope,
    isCurrent: () => io.ownerCurrent,
    verificationState: io.verificationState,
    retryVerification: io.verify,
  }),
}));
vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) => value,
}));
vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: (...args: unknown[]) => {
    io.history(...args);
    return { data: [], total: io.historyTotal };
  },
}));
vi.mock("@/hooks/chat/useClearChat", () => ({ useClearChat: () => vi.fn() }));
vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: { temporaryChat: boolean }) => unknown,
  ) => selector({ temporaryChat: io.temporary }),
}));
vi.mock("@/store/folder", () => ({
  useFolderStore: (
    selector: (state: Record<string, unknown>) => unknown,
  ) => selector(io.folderData),
}));
vi.mock("@/store/route-transition", () => ({
  useRouteTransitionStore: (
    selector: (state: { start: () => void }) => unknown,
  ) => selector({ start: vi.fn() }),
}));
vi.mock("../ServerChatList", () => ({
  ServerChatList: ({
    searchQuery,
    readOnly,
    onConversationSelected,
  }: {
    searchQuery: string;
    readOnly?: boolean;
    onConversationSelected?: () => void;
  }) => (
    <div
      data-testid="server-chat-list"
      data-query={searchQuery}
      data-readonly={String(readOnly)}
    >
      <button onClick={onConversationSelected}>Accept server selection</button>
    </div>
  ),
}));
vi.mock("../LocalChatList", () => ({
  LocalChatList: ({
    searchQuery,
    readOnly,
  }: {
    searchQuery: string;
    readOnly?: boolean;
  }) => (
    <div
      data-testid="local-chat-list"
      data-query={searchQuery}
      data-readonly={String(readOnly)}
    />
  ),
}));
vi.mock("../FolderChatList", () => ({
  FolderChatList: (props: { onConversationSelected?: () => void; readOnly?: boolean }) => {
    io.folderProps(props);
    return <div data-testid="folder-chat-list" data-readonly={String(props.readOnly)}>
      <button onClick={props.onConversationSelected}>Accept folder selection</button>
    </div>;
  },
}));
vi.mock("../../QuickChatHelper", () => ({ QuickChatHelperButton: () => null }));
vi.mock("../../NotesDock", () => ({ NotesDockButton: () => null }));
vi.mock("@/components/Sidepanel/Chat/ModeToggle", () => ({
  ModeToggle: () => null,
}));

const translations = i18next.createInstance().use(ICU);
await translations.init({
  lng: "en",
  fallbackLng: "en",
  resources: { en: { common } },
  interpolation: { escapeValue: false },
});
const recentKey = "tldw:sidebar:recentCollapsed";
const shortcutsKey = "tldw:sidebar:shortcutsCollapsed";
const renderSidebar = (
  props: Partial<React.ComponentProps<typeof ChatSidebar>> = {},
  translator = translations,
) =>
  render(
    <I18nextProvider i18n={translator}>
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatSidebar collapsed={false} {...props} />
      </MemoryRouter>
    </I18nextProvider>,
  );
const recentToggle = () =>
  screen.getByRole("button", { name: "Recent conversations" });
const findInput = () =>
  screen.getByRole("textbox", { name: "Find conversations" });
const preferenceWrites = () =>
  io.writes.mock.calls.filter(
    ([key]) => key === recentKey || key === shortcutsKey,
  );

describe("history-first sidebar navigation", () => {
  beforeEach(() => {
    // This suite seeds the in-memory storage mock directly (no setSetting, and
    // jsdom has no storage.onChanged), so the registry's module-level value
    // cache must be dropped or reads serve the previous test's settings.
    __clearSettingCachesForTests();
    io.values.clear();
    io.reads.clear();
    io.writes.mockClear();
    io.history.mockClear();
    io.historyTotal = 0;
    io.temporary = false;
    io.ownerCurrent = true; io.verificationState = "ready";
    io.folderProps.mockClear(); io.verify.mockClear();
    io.folderData = {conversationKeywordLinks: [], serverDataScope: null};
    io.values.set("tldw:sidebar:shortcutSelection", ["quick-ingest", "chat"]);
    useChatSurfaceCoordinatorStore.setState({
      routeId: null,
      surface: null,
      visiblePanels: {
        "server-history": false,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false,
      },
      engagedPanels: {
        "server-history": false,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false,
      },
    });
  });

  // AX-15 (#3124): the rail and panel toggles only flipped their labels.
  it.each([
    [false, "true", "Collapse sidebar"],
    [true, "false", "Expand sidebar"],
  ] as const)(
    "exposes whether the sidebar is expanded (collapsed=%s)",
    (collapsed, expanded, name) => {
      renderSidebar({ collapsed });

      const toggle = screen.getByTestId("chat-sidebar-toggle");
      expect(toggle).toHaveAccessibleName(name);
      expect(toggle).toHaveAttribute("aria-expanded", expanded);
      // The toggle names the sidebar it opens and closes.
      const controlled = toggle.getAttribute("aria-controls");
      expect(controlled).toBeTruthy();
      expect(document.getElementById(String(controlled))).toBe(
        screen.getByTestId("chat-sidebar"),
      );
    },
  );

  it("names the global Server total while searching both sources", async () => {
    io.historyTotal = 60;
    renderSidebar();
    await screen.findByRole("radio", { name: "Server (60 total)" });
    fireEvent.change(findInput(), { target: { value: "needle" } });
    expect(screen.getByRole("radio", { name: "Server (60 total)" })).toBeInTheDocument();
    expect(screen.getByTestId("server-chat-list")).toHaveAttribute("data-query", "needle");
    expect(screen.getByTestId("local-chat-list")).toHaveAttribute("data-query", "needle");
  });

  it.each(["zero", "unverified", "retired"])("omits an unavailable Server total (%s)", async (state) => {
    io.historyTotal = state === "zero" ? 0 : 60;
    if (state === "unverified") io.verificationState = "error";
    if (state === "retired") io.ownerCurrent = false;
    renderSidebar();
    expect(await screen.findByRole("radio", { name: "Server", exact: true })).toBeInTheDocument();
  });

  it.each([{ count: 1, label: "Folders (1 chat)" }, { count: 2, label: "Folders (2 chats)" }])(
    "names $count unique chats across folders with actual ICU", async ({ count, label }) => {
      io.folderData = { folders: [{id: 1, deleted: false}, {id: 2, deleted: false}], keywords: [], folderKeywordLinks: [],
        conversationKeywordLinks: [], getConversationsForFolder: () => count === 1 ? ["a", "a"] : ["a", "a", "b"],
        serverDataScope: {requestScope: io.requestScope, isCurrent: () => true},
        isLoading: false, error: null, folderApiAvailable: true };
      io.values.set("tldw:sidebar:activeTab", "folders");
      renderSidebar();
      expect(await screen.findByRole("radio", { name: label })).toBeInTheDocument();
    },
  );

  it("formats missing count keys through the component ICU fallback", async () => {
    const missingKeys = structuredClone(common);
    delete (missingKeys.chatSidebar.tabs as Record<string, unknown>).serverTotal;
    delete (missingKeys.chatSidebar.tabs as Record<string, unknown>).foldersChats;
    const missing = i18next.createInstance().use(ICU);
    await missing.init({ lng: "en", fallbackLng: false, resources: {en: {common: missingKeys}}, interpolation: {escapeValue: false} });
    io.historyTotal = 60;
    io.folderData = { folders: [{id: 1, deleted: false}], keywords: [], folderKeywordLinks: [], conversationKeywordLinks: [],
      getConversationsForFolder: () => ["a"], serverDataScope: {requestScope: io.requestScope, isCurrent: () => true},
      isLoading: false, error: null, folderApiAvailable: true };
    renderSidebar({}, missing);
    expect(await screen.findByRole("radio", { name: "Server (60 total)" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Folders (1 chat)" })).toBeInTheDocument();
  });

  it("uses All on first use so one query reaches Server and On this device without replacing an explicit preference", async () => {
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    expect(screen.getByTestId("local-chat-list")).toBeInTheDocument();
    expect(
      screen.getByText("Search Server and On this device history."),
    ).toBeVisible();
    fireEvent.change(findInput(), { target: { value: "needle" } });
    expect(screen.getByTestId("server-chat-list")).toHaveAttribute(
      "data-query",
      "needle",
    );
    expect(screen.getByTestId("local-chat-list")).toHaveAttribute(
      "data-query",
      "needle",
    );
    expect(
      io.writes.mock.calls.filter(([key]) => key === "tldw:sidebar:activeTab"),
    ).toEqual([]);
  });

  it("shows Find conversations and expanded recent history before platform shortcuts on first use", async () => {
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    expect(recentToggle()).toHaveAttribute("aria-expanded", "true");
    const shortcuts = screen.getByRole("button", { name: "Navigation" });
    expect(
      findInput().compareDocumentPosition(recentToggle()) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      recentToggle().compareDocumentPosition(shortcuts) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(preferenceWrites()).toEqual([]);
  });

  it("does not mount history while the saved expansion preference is still being read", async () => {
    let resolve!: (value: unknown) => void;
    const pending = new Promise<unknown>((done) => {
      resolve = done;
    });
    io.reads.set(recentKey, () => pending);
    renderSidebar();
    expect(recentToggle()).toBeDisabled();
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
    expect(io.history).toHaveBeenLastCalledWith(
      "",
      expect.objectContaining({ enabled: false }),
    );
    await act(async () => {
      resolve(true);
    });
    expect(recentToggle()).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
    expect(findInput()).toBeVisible();
  });

  it("waits for a saved on-device source before mounting any history", async () => {
    let resolve!: (value: unknown) => void;
    const pending = new Promise<unknown>((done) => {
      resolve = done;
    });
    io.reads.set("tldw:sidebar:activeTab", () => pending);
    renderSidebar();
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
    expect(findInput()).toBeDisabled();
    expect(io.history).toHaveBeenLastCalledWith(
      "",
      expect.objectContaining({ enabled: false }),
    );
    await act(async () => {
      resolve("local");
    });
    expect(await screen.findByTestId("local-chat-list")).toBeInTheDocument();
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
  });

  it("writes an explicit collapse once and restores it after remount", async () => {
    const first = renderSidebar();
    await screen.findByTestId("server-chat-list");
    fireEvent.click(recentToggle());
    await waitFor(() => expect(io.values.get(recentKey)).toBe(true));
    expect(await getSetting(settings.SIDEBAR_RECENT_COLLAPSED_SETTING)).toBe(
      true,
    );
    first.unmount();
    renderSidebar();
    await waitFor(() =>
      expect(recentToggle()).toHaveAttribute("aria-expanded", "false"),
    );
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
    expect(preferenceWrites()).toEqual([[recentKey, true]]);
  });

  it.each([true, false])(
    "preserves explicit history/shortcut choices through close, reopen and openResetKey (%s)",
    async (closed) => {
      io.values.set(recentKey, closed);
      io.values.set(shortcutsKey, closed);
      const view = renderSidebar({ openResetKey: 1 });
      await waitFor(() =>
        expect(recentToggle()).toHaveAttribute(
          "aria-expanded",
          String(!closed),
        ),
      );
      const shell = (collapsed: boolean, key: number) => (
        <I18nextProvider i18n={translations}>
          <MemoryRouter>
            <ChatSidebar collapsed={collapsed} openResetKey={key} />
          </MemoryRouter>
        </I18nextProvider>
      );
      view.rerender(shell(true, 1));
      view.rerender(shell(false, 2));
      expect(recentToggle()).toHaveAttribute("aria-expanded", String(!closed));
      expect(screen.getByRole("button", { name: "Navigation" })).toHaveAttribute(
        "aria-expanded",
        String(!closed),
      );
      expect(preferenceWrites()).toEqual([]);
    },
  );

  it("does not replace explicit expansion preferences when navigating to another page", async () => {
    io.values.set(recentKey, true);
    io.values.set(shortcutsKey, false);
    const Harness = () => {
      const navigate = useNavigate();
      return (
        <>
          <ChatSidebar />
          <button onClick={() => navigate("/settings")}>Change page</button>
        </>
      );
    };
    render(
      <I18nextProvider i18n={translations}>
        <MemoryRouter>
          <Harness />
        </MemoryRouter>
      </I18nextProvider>,
    );
    await waitFor(() =>
      expect(recentToggle()).toHaveAttribute("aria-expanded", "false"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Change page" }));
    expect(screen.getByRole("button", { name: "Navigation" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(recentToggle()).toHaveAttribute("aria-expanded", "false");
    expect(preferenceWrites()).toEqual([]);
  });

  it("finds chats with history collapsed without overwriting the saved expansion choice", async () => {
    io.values.set(recentKey, true);
    renderSidebar();
    await waitFor(() =>
      expect(recentToggle()).toHaveAttribute("aria-expanded", "false"),
    );
    fireEvent.change(findInput(), { target: { value: "alpha" } });
    expect(screen.getByTestId("server-chat-list")).toBeInTheDocument();
    fireEvent.change(findInput(), { target: { value: "" } });
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
    expect(preferenceWrites()).toEqual([]);
  });

  it("collapses an active search explicitly and retains that choice", async () => {
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    fireEvent.change(findInput(), { target: { value: "alpha" } });
    fireEvent.click(recentToggle());
    expect(findInput()).toHaveValue("");
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
    await waitFor(() => expect(io.values.get(recentKey)).toBe(true));
  });

  it("opens the collapsed rail and focuses Find conversations", async () => {
    const Harness = () => {
      const [collapsed, setCollapsed] = React.useState(true);
      return (
        <ChatSidebar
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsed(false)}
        />
      );
    };
    render(
      <I18nextProvider i18n={translations}>
        <MemoryRouter>
          <Harness />
        </MemoryRouter>
      </I18nextProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Find conversations" }));
    await waitFor(() => expect(findInput()).toHaveFocus());
    expect(preferenceWrites()).toEqual([]);
  });

  it("retains the collapsed-rail Find focus request until the saved source is ready", async () => {
    let resolve!: (value: unknown) => void;
    const pending = new Promise<unknown>((done) => {
      resolve = done;
    });
    io.reads.set("tldw:sidebar:activeTab", () => pending);
    const Harness = () => {
      const [collapsed, setCollapsed] = React.useState(true);
      return (
        <ChatSidebar
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsed(false)}
        />
      );
    };
    render(
      <I18nextProvider i18n={translations}>
        <MemoryRouter>
          <Harness />
        </MemoryRouter>
      </I18nextProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Find conversations" }));
    expect(findInput()).toBeDisabled();
    await act(async () => {
      resolve("local");
    });
    await waitFor(() => expect(findInput()).toHaveFocus());
    expect(screen.queryByTestId("server-chat-list")).toBeNull();
  });

  it.each(["new control", "close and reopen", "route replacement"])(
    "cancels delayed Find focus after %s",
    async (changed) => {
      let resolve!: (value: unknown) => void;
      const pending = new Promise<unknown>((done) => {
        resolve = done;
      });
      io.reads.set("tldw:sidebar:activeTab", () => pending);
      const Harness = () => {
        const [collapsed, setCollapsed] = React.useState(true);
        const navigate = useNavigate();
        return (
          <>
            <ChatSidebar
              collapsed={collapsed}
              onToggleCollapse={() => setCollapsed((value) => !value)}
            />
            <button onClick={() => navigate("/settings")}>Replace route</button>
          </>
        );
      };
      render(
        <I18nextProvider i18n={translations}>
          <MemoryRouter initialEntries={["/chat"]}>
            <Harness />
          </MemoryRouter>
        </I18nextProvider>,
      );
      fireEvent.click(
        screen.getByRole("button", { name: "Find conversations" }),
      );
      expect(findInput()).toBeDisabled();
      if (changed === "new control") {
        const next = screen.getByRole("button", { name: "New Chat" });
        next.focus();
        fireEvent.click(next);
      } else if (changed === "close and reopen") {
        fireEvent.click(
          screen.getByRole("button", { name: "Collapse sidebar" }),
        );
        fireEvent.click(screen.getByRole("button", { name: "Expand sidebar" }));
      } else {
        fireEvent.click(screen.getByRole("button", { name: "Replace route" }));
      }
      await act(async () => {
        resolve("local");
      });
      expect(findInput()).not.toHaveFocus();
      if (changed === "new control")
        expect(screen.getByRole("button", { name: "New Chat" })).toHaveFocus();
    },
  );

  it("keeps search text when switching between server and on-device history", async () => {
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    fireEvent.change(findInput(), { target: { value: "alpha" } });
    fireEvent.click(screen.getByText("On this device"));
    expect(await screen.findByTestId("local-chat-list")).toBeInTheDocument();
    expect(findInput()).toHaveValue("alpha");
    expect(screen.getByText("Search On this device history.")).toBeVisible();
    expect(io.history).toHaveBeenLastCalledWith(
      "",
      expect.objectContaining({ enabled: false }),
    );
  });

  it("explains and performs server history search when starting from Folders", async () => {
    io.values.set("tldw:sidebar:activeTab", "folders");
    renderSidebar();
    await screen.findByTestId("folder-chat-list");
    expect(
      screen.getByText("Typing searches Server and On this device history."),
    ).toBeVisible();
    fireEvent.change(findInput(), { target: { value: "alpha" } });
    expect(await screen.findByTestId("server-chat-list")).toBeInTheDocument();
    expect(screen.queryByTestId("folder-chat-list")).toBeNull();
    expect(screen.getByTestId("local-chat-list")).toHaveAttribute(
      "data-query",
      "alpha",
    );
    fireEvent.change(findInput(), { target: { value: "" } });
    await waitFor(() =>
      expect(screen.getByTestId("folder-chat-list")).toBeInTheDocument(),
    );
    expect(io.values.get("tldw:sidebar:activeTab")).toBe("folders");
    expect(
      io.writes.mock.calls.filter(([key]) => key === "tldw:sidebar:activeTab"),
    ).toEqual([]);
    expect(findInput()).toHaveValue("");
  });

  it.each(["current", "unqualified", "retired", "failed", "unverified"])(
    "qualifies the Folders badge to a successful current parent snapshot (%s)", async (state) => {
      const operation = {requestScope: io.requestScope, isCurrent: () => state !== "retired"};
      io.folderData = { folders: [{id: 1, deleted: false}], keywords: [], folderKeywordLinks: [],
        conversationKeywordLinks: [{conversation_id: "a"}, {conversation_id: "b"}],
        getConversationsForFolder: () => ["a", "a", "b"],
        serverDataScope: state === "unqualified" ? null : operation,
        isLoading: false, error: state === "failed" ? "Unavailable" : null, folderApiAvailable: true };
      if (state === "unverified") io.verificationState = "error";
      io.values.set("tldw:sidebar:activeTab", "folders"); renderSidebar();
      await screen.findByTestId("folder-chat-list");
      expect(screen.queryByText("Folders (2 chats)") !== null).toBe(state === "current");
    },
  );

  it("forwards the verified parent owner and Temporary read-only guard to Folders", async () => {
    io.temporary = true; io.values.set("tldw:sidebar:activeTab", "folders"); renderSidebar();
    expect(await screen.findByTestId("folder-chat-list")).toHaveAttribute("data-readonly", "true");
    const props = io.folderProps.mock.calls.at(-1)![0];
    expect(props.owner.requestScope).toBe(io.requestScope);
    expect(props.verificationState).toBe("ready");
    expect(props.onRetryVerification).toBe(io.verify);
    expect(screen.getByTestId("folder-chat-list").closest(".pointer-events-none")).toBeNull();
  });

  it("keeps all custom shortcuts below history in one scroller with search fixed above", async () => {
    io.values.set("tldw:sidebar:shortcutSelection", [
      "quick-ingest",
      "kanban-playground",
      "quick-ingest",
      "chat",
    ]);
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    const shortcuts = within(
      document.getElementById("chat-sidebar-shortcuts")!,
    ).getAllByRole("button");
    expect(shortcuts.map((button) => button.dataset.testid)).toEqual([
      "chat-sidebar-shortcut-quick-ingest",
      "chat-sidebar-shortcut-kanban-playground",
      "chat-sidebar-shortcut-chat",
    ]);
    const scroller = recentToggle().closest(".overflow-y-auto");
    expect(scroller).not.toContainElement(findInput());
    expect(scroller).toContainElement(recentToggle());
    expect(scroller).toContainElement(shortcuts[0]);
    expect(scroller).not.toContainElement(
      screen.getByTestId("chat-sidebar-settings"),
    );
    expect(
      screen.getByTestId("chat-sidebar").querySelectorAll(".overflow-y-auto"),
    ).toHaveLength(1);
  });

  it.each(["server", "folders"])(
    "forwards only accepted %s selection callbacks",
    async (tab) => {
      io.values.set("tldw:sidebar:activeTab", tab);
      const accepted = vi.fn();
      renderSidebar({ onConversationSelected: accepted });
      const action = await screen.findByRole("button", {
        name:
          tab === "server"
            ? "Accept server selection"
            : "Accept folder selection",
      });
      expect(accepted).not.toHaveBeenCalled();
      fireEvent.click(action);
      expect(accepted).toHaveBeenCalledOnce();
    },
  );

  it("exits server multi-selection when history collapses", async () => {
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    fireEvent.click(screen.getByRole("button", { name: "Select chats" }));
    expect(
      screen.getByRole("button", { name: "Exit selection" }),
    ).toBeInTheDocument();
    fireEvent.click(recentToggle());
    fireEvent.click(recentToggle());
    expect(
      screen.getByRole("button", { name: "Select chats" }),
    ).toBeInTheDocument();
  });

  it("keeps Temporary history readonly and the on-device view lazy from server requests", async () => {
    io.temporary = true;
    io.values.set("tldw:sidebar:activeTab", "local");
    renderSidebar();
    expect(await screen.findByTestId("local-chat-list")).toHaveAttribute(
      "data-readonly",
      "true",
    );
    expect(io.history).toHaveBeenLastCalledWith(
      "",
      expect.objectContaining({ enabled: false }),
    );
    expect(screen.queryByRole("button", { name: "Select chats" })).toBeNull();
  });
});
