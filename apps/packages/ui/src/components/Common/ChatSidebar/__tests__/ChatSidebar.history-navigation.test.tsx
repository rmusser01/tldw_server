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
import { I18nextProvider } from "react-i18next";
import common from "@/assets/locale/en/common.json";
import { ChatSidebar } from "../../ChatSidebar";
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator";
import * as settings from "@/services/settings/ui-settings";
import { getSetting } from "@/services/settings/registry";

type Change = { newValue: unknown; oldValue: unknown };
const io = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  listeners: new Map<string, Set<(change: Change) => void>>(),
  reads: new Map<string, () => Promise<unknown>>(),
  writes: vi.fn(),
  history: vi.fn(),
  temporary: false,
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
      return true;
    },
    unwatch: (listeners: Record<string, (change: Change) => void>) => {
      for (const [key, listener] of Object.entries(listeners))
        io.listeners.get(key)?.delete(listener);
    },
  }),
}));
vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) => value,
}));
vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: (...args: unknown[]) => {
    io.history(...args);
    return { data: [], total: 0 };
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
    selector: (state: { conversationKeywordLinks: never[] }) => unknown,
  ) => selector({ conversationKeywordLinks: [] }),
}));
vi.mock("@/store/route-transition", () => ({
  useRouteTransitionStore: (
    selector: (state: { start: () => void }) => unknown,
  ) => selector({ start: vi.fn() }),
}));
vi.mock("../ServerChatList", () => ({
  ServerChatList: ({
    readOnly,
    onConversationSelected,
  }: {
    readOnly?: boolean;
    onConversationSelected?: () => void;
  }) => (
    <div data-testid="server-chat-list" data-readonly={String(readOnly)}>
      <button onClick={onConversationSelected}>Accept server selection</button>
    </div>
  ),
}));
vi.mock("../LocalChatList", () => ({
  LocalChatList: ({ readOnly }: { readOnly?: boolean }) => (
    <div data-testid="local-chat-list" data-readonly={String(readOnly)} />
  ),
}));
vi.mock("../FolderChatList", () => ({
  FolderChatList: ({
    onConversationSelected,
  }: {
    onConversationSelected?: () => void;
  }) => (
    <div data-testid="folder-chat-list">
      <button onClick={onConversationSelected}>Accept folder selection</button>
    </div>
  ),
}));
vi.mock("../../QuickChatHelper", () => ({ QuickChatHelperButton: () => null }));
vi.mock("../../NotesDock", () => ({ NotesDockButton: () => null }));
vi.mock("@/components/Sidepanel/Chat/ModeToggle", () => ({
  ModeToggle: () => null,
}));

const translations = i18next.createInstance();
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
) =>
  render(
    <I18nextProvider i18n={translations}>
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
    io.values.clear();
    io.reads.clear();
    io.writes.mockClear();
    io.history.mockClear();
    io.temporary = false;
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

  it("shows Find conversations and expanded recent history before platform shortcuts on first use", async () => {
    renderSidebar();
    await screen.findByTestId("server-chat-list");
    expect(recentToggle()).toHaveAttribute("aria-expanded", "true");
    const shortcuts = screen.getByRole("button", { name: "Shortcuts" });
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
      expect(screen.getByRole("button", { name: "Shortcuts" })).toHaveAttribute(
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
    expect(screen.getByRole("button", { name: "Shortcuts" })).toHaveAttribute(
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
    expect(screen.getByText("Typing searches Server history.")).toBeVisible();
    fireEvent.change(findInput(), { target: { value: "alpha" } });
    expect(await screen.findByTestId("server-chat-list")).toBeInTheDocument();
    expect(screen.queryByTestId("folder-chat-list")).toBeNull();
    expect(findInput()).toHaveValue("alpha");
  });

  it("keeps all custom shortcuts in their saved order below history in one scroller", async () => {
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
    const scroller = findInput().closest(".overflow-y-auto");
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
