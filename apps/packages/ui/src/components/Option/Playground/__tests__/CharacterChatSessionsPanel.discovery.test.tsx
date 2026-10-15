import React from "react";
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CharacterChatSessionsPanel } from "../CharacterChatSessionsPanel";
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner";
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient";

const io = vi.hoisted(() => ({
  list: vi.fn(),
  initialize: vi.fn(),
  select: vi.fn(),
  resolve: vi.fn(),
  check: vi.fn(),
  account: "A",
  connected: true,
  listeners: new Set<() => void>(),
  accountListeners: new Set<(invalidated: boolean) => void>(),
}));
const scope = () => ({
  userId: io.account,
  scopeKey: `server:${io.account}`,
  clientPrincipalVerified: true,
  config: {
    serverUrl: "https://chat.test",
    authMode: "multi-user" as const,
    apiKey: "private-test-key",
  },
});
vi.mock("@/services/service-prompts", async (original) => ({
  ...(await original<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: () => io.resolve(),
}));
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (cb: (invalidated: boolean) => void) => {
    io.accountListeners.add(cb);
    return () => io.accountListeners.delete(cb);
  },
}));
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    isConnected: React.useSyncExternalStore(
      (cb) => {
        io.listeners.add(cb);
        return () => io.listeners.delete(cb);
      },
      () => io.connected,
    ),
  }),
}));
vi.mock("@/store/connection", () => ({
  useConnectionStore: Object.assign(
    (selector: (state: unknown) => unknown) =>
      selector({ checkOnce: io.check }),
    { getState: () => ({ state: { isConnected: io.connected } }) },
  ),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => io.initialize(...args),
    listChatsWithMeta: (...args: unknown[]) => io.list(...args),
  },
}));
vi.mock("@/hooks/chat/useSelectServerChat", () => ({
  useSelectServerChat: () => io.select,
}));
vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next");
  const { default: playground } =
    await import("@/assets/locale/en/playground.json");
  const { default: common } = await import("@/assets/locale/en/common.json");
  const instance = createInstance();
  await instance.init({
    lng: "en",
    fallbackLng: "en",
    defaultNS: "playground",
    resources: { en: { playground, common } },
    interpolation: { escapeValue: false },
  });
  return { useTranslation: () => ({ t: instance.t.bind(instance) }) };
});

const chat = (
  id: string,
  characterId = 7,
  overrides: Partial<ServerChatSummary> = {},
): ServerChatSummary => ({
  id,
  title: `Session ${id}`,
  character_id: characterId,
  character_name: characterId === 7 ? "Mira" : "Rook",
  created_at: "2026-05-20T00:00:00.000Z",
  updated_at: "2026-05-20T00:10:00.000Z",
  ...overrides,
});
let rows: ServerChatSummary[];
const makeClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
const mount = (
  props: React.ComponentProps<typeof CharacterChatSessionsPanel> = {},
) => {
  const client = makeClient();
  const renderPanel = (next = props) => (
    <QueryClientProvider client={client}>
      <CharacterChatSessionsPanel
        activeCharacterId={7}
        activeCharacterName="Mira"
        {...next}
      />
    </QueryClientProvider>
  );
  const rendered = render(renderPanel());
  return {
    ...rendered,
    client,
    rerenderPanel: (next: typeof props) => rendered.rerender(renderPanel(next)),
  };
};
const setAccount = (account: string) =>
  act(() => {
    io.account = account;
    for (const cb of io.accountListeners) cb(true);
  });
const setConnected = (connected: boolean) =>
  act(() => {
    io.connected = connected;
    for (const cb of io.listeners) cb();
  });
const fullView = async () => {
  fireEvent.click(
    await screen.findByRole("button", {
      name: "All sessions for this character",
    }),
  );
  await screen.findByRole("button", { name: "Back to recent sessions" });
};

beforeEach(() => {
  vi.clearAllMocks();
  io.account = "A";
  io.connected = true;
  io.resolve.mockImplementation(async () => scope());
  io.initialize.mockResolvedValue(undefined);
  io.check.mockResolvedValue(undefined);
  rows = [
    ...Array.from({ length: 8 }, (_, i) => chat(`other-${i}`, 9)),
    ...Array.from({ length: 32 }, (_, i) => chat(`mira-${i}`)),
  ];
  io.list.mockImplementation(async (params) => {
    const filtered =
      params.character_id == null
        ? rows
        : rows.filter(
            (row) => Number(row.character_id) === params.character_id,
          );
    return {
      chats: filtered.slice(params.offset, params.offset + params.limit),
      total: filtered.length,
    };
  });
});

describe("owned complete character-session discovery", () => {
  it("filters the active character before sampling and scopes the request to the verified account", async () => {
    mount();
    expect(await screen.findByText("Session mira-0")).toBeInTheDocument();
    expect(io.list).toHaveBeenCalledWith(
      expect.objectContaining({ character_id: 7, limit: 5, offset: 0 }),
      expect.objectContaining({
        requestScope: expect.objectContaining({ userId: "A" }),
      }),
    );
    expect(screen.queryByText("Session other-0")).not.toBeInTheDocument();
    expect(io.list).toHaveBeenCalledTimes(1);
  });

  it("reaches older pages and preserves distinct conversations with duplicate titles", async () => {
    rows[8].title = "Campfire";
    rows[9].title = "Campfire";
    mount();
    await fullView();
    expect(
      await screen.findAllByRole("button", { name: /^Resume Campfire/ }),
    ).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Next sessions page" }));
    expect(await screen.findByText("Session mira-31")).toBeInTheDocument();
    expect(screen.queryByText("Campfire")).not.toBeInTheDocument();
    expect(io.list).toHaveBeenCalledWith(
      expect.objectContaining({ character_id: 7, offset: 25, limit: 25 }),
      expect.anything(),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Previous sessions page" }),
    );
    const duplicateActions = await screen.findAllByRole("button", {
      name: /^Resume Campfire/,
    });
    expect(
      duplicateActions.map((button) => button.getAttribute("aria-label")),
    ).toEqual([
      "Resume Campfire, session mira-0",
      "Resume Campfire, session mira-1",
    ]);
    expect(screen.getByText("Session mira-0")).toBeVisible();
    expect(screen.getByText("Session mira-1")).toBeVisible();
    fireEvent.click(duplicateActions[1]);
    expect(io.select).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ id: "mira-1" }),
    );
  });

  it("offers next for a full page even when the server count falls back to page length", async () => {
    io.list.mockImplementation(async (params) => ({
      chats: rows
        .filter((row) => row.character_id === 7)
        .slice(params.offset, params.offset + params.limit),
      total: params.limit,
    }));
    mount();
    await fullView();
    await screen.findByText("Session mira-24");
    fireEvent.click(screen.getByRole("button", { name: "Next sessions page" }));
    expect(await screen.findByText("Session mira-31")).toBeInTheDocument();
  });

  it("labels a later empty page without falsely declaring no conversations and retains Previous", async () => {
    rows = rows.filter((row) => row.character_id === 7).slice(0, 25);
    mount();
    await fullView();
    await screen.findByText("Session mira-24");
    fireEvent.click(screen.getByRole("button", { name: "Next sessions page" }));
    expect(
      await screen.findByText(
        "No sessions on this page. Return to the previous page to see earlier results.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("No conversations with this character yet."),
    ).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Previous sessions page" }),
    );
    expect(await screen.findByText("Session mira-24")).toBeInTheDocument();
  });

  it("keeps other-character recents lazy and labels them as a bounded sample", async () => {
    mount();
    await screen.findByText("Session mira-0");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Recent chats with other characters",
      }),
    );
    expect(await screen.findByText("Session other-0")).toBeInTheDocument();
    expect(
      screen.getByText(
        "A sample from the 5 most recent character chats; older chats may not appear here.",
      ),
    ).toBeInTheDocument();
    expect(io.list).toHaveBeenCalledWith(
      expect.not.objectContaining({ character_id: expect.anything() }),
      expect.anything(),
    );
  });

  it("resume-last belongs to the selected character even when other characters are more recent", async () => {
    mount();
    fireEvent.click(
      await screen.findByRole("button", {
        name: "Resume last character chat: Session mira-0",
      }),
    );
    expect(io.select).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ id: "mira-0", character_id: 7 }),
    );
  });

  it("retains current-row disabling, session metadata and normalized date", async () => {
    rows = [
      chat("current", 7, {
        title: "Campfire",
        updated_at: " 2026-05-20T00:10:00.000Z ",
        topic_label: "Act I",
        message_count: 6,
      }),
    ];
    mount({ activeServerChatId: "current" });
    expect(
      await screen.findByRole("button", { name: "Current Campfire" }),
    ).toBeDisabled();
    const row = screen.getByText("Campfire").closest("li")!;
    expect(within(row).getByText("Mira")).toBeInTheDocument();
    expect(row).toHaveTextContent("Act I");
    expect(row).toHaveTextContent("6 messages");
    expect(row).toHaveTextContent("Saved");
    expect(row.querySelector("time")?.dateTime).toBe(
      "2026-05-20T00:10:00.000Z",
    );
    expect(
      screen.queryByRole("button", { name: /Resume last/ }),
    ).not.toBeInTheDocument();
  });

  it("recognizes character assistant identity but rejects persona rows in the filtered result", async () => {
    io.list.mockResolvedValue({
      chats: [
        chat("assistant", 7, {
          character_id: null,
          assistant_kind: "character",
          assistant_id: "7",
        }),
        chat("persona", 7, {
          character_id: null,
          assistant_kind: "persona",
          assistant_id: "7",
        }),
      ],
      total: 2,
    });
    mount();
    expect(await screen.findByText("Session assistant")).toBeInTheDocument();
    expect(screen.queryByText("Session persona")).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "Some sessions in this page could not be verified for this character. Retry to refresh.",
      ),
    ).toBeInTheDocument();
  });

  it("rejects a foreign workspace row returned to the global character panel", async () => {
    io.list.mockResolvedValue({
      chats: [
        chat("workspace-secret", 7, {
          scope_type: "workspace",
          workspace_id: "other",
        }),
      ],
      total: 1,
    });
    mount();
    expect(
      await screen.findByText(
        "Some sessions in this page could not be verified for this character. Retry to refresh.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Session workspace-secret"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("No conversations with this character yet."),
    ).not.toBeInTheDocument();
  });

  it("shows verified empty only after a successful selected-character first page", async () => {
    rows = [];
    mount();
    expect(
      await screen.findByText("No conversations with this character yet."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("No character conversations yet."),
    ).not.toBeInTheDocument();
  });

  it("shows loading without an empty or None badge while a page is pending", async () => {
    io.list.mockReturnValue(new Promise(() => {}));
    mount();
    expect(
      await screen.findByText("Loading character sessions..."),
    ).toBeInTheDocument();
    expect(screen.queryByText("None")).not.toBeInTheDocument();
    expect(
      screen.queryByText("No conversations with this character yet."),
    ).not.toBeInTheDocument();
  });

  it.each([429, 500])(
    "failed next-page %s offers retry and previous without a false empty state",
    async (status) => {
      const original = io.list.getMockImplementation()!;
      let fails = true;
      io.list.mockImplementation((params, options) =>
        params.offset === 25 && fails
          ? Promise.reject(Object.assign(new Error("Page failed"), { status }))
          : original(params, options),
      );
      mount();
      await fullView();
      await screen.findByText("Session mira-24");
      fireEvent.click(
        screen.getByRole("button", { name: "Next sessions page" }),
      );
      const retry = await screen.findByRole(
        "button",
        { name: "Retry character sessions" },
        { timeout: 4000 },
      );
      expect(
        screen.queryByText("No conversations with this character yet."),
      ).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Previous sessions page" }),
      ).toBeEnabled();
      fails = false;
      fireEvent.click(retry);
      expect(await screen.findByText("Session mira-31")).toBeInTheDocument();
    },
  );

  it("marks cached results stale after a hard refresh failure and retries the same page", async () => {
    const { client } = mount();
    await screen.findByText("Session mira-0");
    io.list.mockRejectedValue(
      Object.assign(new Error("Server unavailable"), { status: 500 }),
    );
    await act(async () => {
      await client.invalidateQueries({ queryKey: ["serverChatHistory"] });
    });
    expect(
      await screen.findByText(
        "Showing character sessions from the last successful refresh.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Session mira-0")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Retry character sessions" }),
    ).toBeEnabled();
  });

  it("marks cached offline data and disables resume without suggesting a successful empty history", async () => {
    mount();
    await screen.findByText("Session mira-0");
    setConnected(false);
    expect(
      await screen.findByText(
        "Reconnect to load or resume character sessions. Showing the last loaded page.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Resume Session mira-0" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", {
        name: "Resume last character chat: Session mira-0",
      }),
    ).toBeDisabled();
  });

  it("distinguishes account verification pending from an offline connection", async () => {
    io.resolve.mockReturnValue(new Promise(() => {}));
    mount();
    expect(
      await screen.findByText("Verifying your server account..."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Connect to your server to view character sessions."),
    ).not.toBeInTheDocument();
    expect(io.list).not.toHaveBeenCalled();
  });

  it("retries a failed first account verification without blur or reload", async () => {
    io.resolve.mockRejectedValueOnce(
      new Error("Temporary verification failure"),
    );
    mount();
    expect(
      await screen.findByText(
        "Could not verify your server account. Retry or check your connection settings.",
      ),
    ).toBeInTheDocument();
    expect(io.list).not.toHaveBeenCalled();
    fireEvent.click(
      screen.getByRole("button", { name: "Retry account verification" }),
    );
    expect(await screen.findByText("Session mira-0")).toBeInTheDocument();
    expect(io.resolve).toHaveBeenCalledTimes(2);
  });

  it("an offline first visit is unavailable, never an empty-success claim", async () => {
    io.connected = false;
    mount();
    expect(
      await screen.findByText(
        "Connect to your server to view character sessions.",
      ),
    ).toBeInTheDocument();
    expect(io.list).not.toHaveBeenCalled();
    expect(
      screen.queryByText("No conversations with this character yet."),
    ).not.toBeInTheDocument();
  });

  it("a local or invalid card ID never silently fetches all history as that character", async () => {
    mount({ activeCharacterId: "local-card-uuid" });
    expect(
      await screen.findByText(
        "Server sessions are available for characters saved on this server.",
      ),
    ).toBeInTheDocument();
    expect(io.list).not.toHaveBeenCalled();
    expect(
      screen.queryByRole("button", { name: "All sessions for this character" }),
    ).not.toBeInTheDocument();
  });

  it("normalizes a numeric string character ID before the request", async () => {
    mount({ activeCharacterId: " 7 " });
    await screen.findByText("Session mira-0");
    expect(io.list).toHaveBeenCalledWith(
      expect.objectContaining({ character_id: 7 }),
      expect.anything(),
    );
  });

  it("disabled panels do not start either query", async () => {
    mount({ enabled: false });
    await act(async () => {
      await Promise.resolve();
    });
    expect(io.list).not.toHaveBeenCalled();
  });

  it.each(["character", "account"])(
    "resets full-view paging and other recents on %s replacement",
    async (boundary) => {
      const rendered = mount();
      await fullView();
      await screen.findByText("Session mira-24");
      fireEvent.click(
        screen.getByRole("button", { name: "Next sessions page" }),
      );
      await screen.findByText("Session mira-31");
      fireEvent.click(
        screen.getByRole("button", {
          name: "Recent chats with other characters",
        }),
      );
      await screen.findByText("Session other-0");
      if (boundary === "character")
        rendered.rerenderPanel({
          activeCharacterId: 9,
          activeCharacterName: "Rook",
        });
      else setAccount("B");
      expect(
        await screen.findByRole("button", {
          name: "All sessions for this character",
        }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", {
          name: "Recent chats with other characters",
        }),
      ).toHaveAttribute("aria-expanded", "false");
      expect(screen.queryByText("Session mira-31")).not.toBeInTheDocument();
      await waitFor(() =>
        expect(io.list).toHaveBeenCalledWith(
          expect.objectContaining({
            offset: 0,
            limit: 5,
            character_id: boundary === "character" ? 9 : 7,
          }),
          expect.objectContaining({
            scope: { type: "global" },
            requestScope: expect.objectContaining({
              userId: boundary === "account" ? "B" : "A",
            }),
          }),
        ),
      );
    },
  );

  it("fences a held response through A-to-B-to-A account changes", async () => {
    let finish!: (value: unknown) => void;
    io.list.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const { client } = mount();
    await waitFor(() => expect(io.list).toHaveBeenCalledTimes(1));
    setAccount("B");
    await screen.findByText("Session mira-0");
    setAccount("A");
    await screen.findByText("Session mira-0");
    await act(async () => {
      finish({ chats: [chat("late-secret")], total: 1 });
      await Promise.resolve();
    });
    expect(screen.queryByText("Session late-secret")).not.toBeInTheDocument();
    expect(
      JSON.stringify(
        client
          .getQueryCache()
          .getAll()
          .map((query) => query.state.data),
      ),
    ).not.toContain("late-secret");
    expect(
      JSON.stringify(
        client
          .getQueryCache()
          .getAll()
          .map((query) => query.queryKey),
      ),
    ).not.toContain("private-test-key");
  });

  it("revokes a rendered resume callback synchronously when the account is invalidated", async () => {
    mount();
    const button = await screen.findByRole("button", {
      name: "Resume Session mira-0",
    });
    act(() => {
      io.account = "B";
      for (const cb of io.accountListeners) cb(true);
      fireEvent.click(button);
    });
    expect(io.select).not.toHaveBeenCalled();
  });

  it("rejects Resume when connection changes offline before React commits", async () => {
    mount();
    const button = await screen.findByRole("button", {
      name: "Resume Session mira-0",
    });
    act(() => {
      io.connected = false;
      for (const cb of io.listeners) cb();
      fireEvent.click(button);
    });
    expect(io.select).not.toHaveBeenCalled();
  });

  it("a disabled failed panel cannot imperatively refetch through Retry", async () => {
    io.list.mockRejectedValue(
      Object.assign(new Error("Rate limited"), { status: 429 }),
    );
    const rendered = mount();
    const retry = await screen.findByRole("button", {
      name: "Retry character sessions",
    });
    const calls = io.list.mock.calls.length;
    rendered.rerenderPanel({ enabled: false });
    fireEvent.click(retry);
    await act(async () => {
      await Promise.resolve();
    });
    expect(io.list).toHaveBeenCalledTimes(calls);
    expect(retry).toBeDisabled();
  });

  it("does not resume a cached row after an offline transition before rerender", async () => {
    mount();
    const button = await screen.findByRole("button", {
      name: "Resume Session mira-0",
    });
    setConnected(false);
    fireEvent.click(button);
    expect(io.select).not.toHaveBeenCalled();
  });

  it("keeps view-switch focus on its persistent control", async () => {
    mount();
    const button = await screen.findByRole("button", {
      name: "All sessions for this character",
    });
    button.focus();
    fireEvent.click(button);
    expect(
      await screen.findByRole("button", { name: "Back to recent sessions" }),
    ).toHaveFocus();
  });
});

describe("verified draft owner request-scope reuse", () => {
  it("exposes only the current transient request scope and revokes an old lease through ABA", async () => {
    const { result } = renderHook(() => useChatDraftOwner(() => {}));
    await waitFor(() => expect(result.current.ownerKey).not.toBeNull());
    const old = result.current;
    expect(old.requestScope).toMatchObject({
      userId: "A",
      config: { serverUrl: "https://chat.test" },
    });
    setAccount("B");
    await waitFor(() => expect(result.current.requestScope?.userId).toBe("B"));
    setAccount("A");
    await waitFor(() => expect(result.current.requestScope?.userId).toBe("A"));
    expect(old.isCurrent()).toBe(false);
  });
});

it("discovers all six interleaved character histories across a 2000-conversation library", async () => {
  rows = Array.from({ length: 2000 }, (_, index) => chat(`scale-${index}`, 7 + index % 6, {
    character_name: `Character ${7 + index % 6}`,
  }))
  const mounted = mount({ activeCharacterId: 7, activeCharacterName: "Character 7" })
  for (let character = 7; character < 13; character++) {
    if (character !== 7) mounted.rerenderPanel({ activeCharacterId: character, activeCharacterName: `Character ${character}` })
    const selected = rows.filter(row => row.character_id === character)
    await screen.findByText(`Session ${selected[0].id}`)
    await fullView()
    await screen.findByText(`Session ${selected[24].id}`)
    for (let offset = 25; offset < selected.length; offset += 25) {
      fireEvent.click(screen.getByRole("button", { name: "Next sessions page" }))
      await screen.findByText(`Session ${selected[offset].id}`)
    }
    const last = selected.at(-1)!
    fireEvent.click(screen.getByRole("button", { name: `Resume Session ${last.id}` }))
    expect(io.select).toHaveBeenLastCalledWith(expect.objectContaining({ id: last.id, character_id: character }))
    expect(io.list).toHaveBeenCalledWith(expect.objectContaining({ character_id: character, offset: 325, limit: 25 }), expect.objectContaining({ requestScope: expect.objectContaining({ userId: "A" }) }))
  }
  expect(io.select).toHaveBeenCalledTimes(6)
  mounted.unmount()
}, 20000)
