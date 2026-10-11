// @vitest-environment jsdom
/**
 * XS-06 (#3105): the side-panel chat list shows recent chats, from this device
 * and the server, without a search. Each opens through the panel's own open
 * handlers, which give it its own tab (XS-01).
 */
import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidepanelChatSidebar } from "../Sidebar"
import type { HistoryInfo } from "@/db/dexie/types"
import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import type { SidepanelChatTab } from "@/store/sidepanel-chat-tabs"

const io = vi.hoisted(() => ({
  listeners: new Set<(profileChanged: boolean) => void>(),
  serverHistory: vi.fn(),
  recentLocal: vi.fn(),
  readProfile: vi.fn(),
  search: vi.fn(async () => [] as unknown[])
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue || key
  })
}))
vi.mock("antd", () => ({
  message: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Modal: ({ open, children }: { open?: boolean; children?: React.ReactNode }) =>
    open ? <div>{children}</div> : null
}))
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))
vi.mock("@/store/sidepanel-chat-tabs", () => ({
  useSidepanelChatTabsStore: (selector?: (state: Record<string, unknown>) => unknown) =>
    typeof selector === "function"
      ? selector({ togglePinned: vi.fn(), renameTab: vi.fn(), setStatus: vi.fn() })
      : {}
}))
vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector?: (state: { mode: string }) => unknown) =>
    typeof selector === 'function' ? selector({ mode: 'pro' }) : { mode: 'pro' },
}));
vi.mock('@/hooks/useDebounce', () => ({ useDebounce: <T,>(value: T) => value }));
vi.mock('@/hooks/useConnectionState', () => ({
  useConnectionState: () => ({ isConnected: true }),
}));
vi.mock('@/hooks/useServerChatHistory', () => ({
  useServerChatHistory: (...args: unknown[]) => io.serverHistory(...args),
}));
vi.mock('@/db/dexie/chat', async () => ({
  ...(await vi.importActual<typeof import('@/db/dexie/history-selection')>(
    '@/db/dexie/history-selection'
  )),
  watchOnDeviceHistoryChanges: (listener: (profileChanged: boolean) => void) => {
    io.listeners.add(listener);
    return () => {
      io.listeners.delete(listener);
    };
  },
  PageAssistDatabase: vi.fn(function MockPageAssistDatabase(this: Record<string, unknown>) {
    this.fullTextSearchChatHistories = io.search;
    this.getRecentChatHistories = io.recentLocal;
  }),
}));
vi.mock('@/db/dexie/schema', () => ({
  db: { userSettings: { get: (...args: unknown[]) => io.readProfile(...args) } },
}));
vi.mock('@/store/folder', () => {
  const state = { getFoldersForConversation: () => [], uiPrefs: {}, folderApiAvailable: true };
  return {
    useFolderStore: (selector?: (value: typeof state) => unknown) =>
      typeof selector === "function" ? selector(state) : state
  }
})
vi.mock("@/hooks/useBulkChatOperations", () => ({
  useBulkChatOperations: () => ({
    openBulkFolderPicker: vi.fn(),
    openBulkTagPicker: vi.fn(),
    applyBulkDelete: vi.fn()
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [288, vi.fn()] }))
vi.mock("../ModeToggle", () => ({ ModeToggle: () => null }))
vi.mock("../ConversationContextMenu", () => ({
  ConversationContextMenu: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))
vi.mock("../FolderPickerModal", () => ({ FolderPickerModal: () => null }))
vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({ showUndoNotification: vi.fn() })
}))

const ownerKey = "alice"
const owner = {
  ownerKey,
  isCurrent: () => true,
  snapshot: {
    requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" as const }, userId: 1 }
  }
}

const serverChat = (id: string, title: string, updatedAtMs: number): ServerChatHistoryItem =>
  ({
    id,
    title,
    created_at: new Date(updatedAtMs).toISOString(),
    updated_at: new Date(updatedAtMs).toISOString(),
    createdAtMs: updatedAtMs,
    updatedAtMs
  }) as ServerChatHistoryItem
const localHistory = (id: string, title: string, createdAt: number, extra: Partial<HistoryInfo> = {}): HistoryInfo => ({
  id,
  title,
  createdAt,
  is_rag: false,
  server_scope_key: ownerKey,
  ...extra
})
const openTab = (id: string, extra: Partial<SidepanelChatTab>): SidepanelChatTab => ({
  id,
  label: `Tab ${id}`,
  historyId: null,
  serverChatId: null,
  serverChatTopic: null,
  updatedAt: Date.now(),
  ...extra
})

const renderSidebar = (props: Partial<React.ComponentProps<typeof SidepanelChatSidebar>> = {}) => {
  const onOpenServerChat = vi.fn()
  const onOpenLocalHistory = vi.fn()
  const view = render(
    <SidepanelChatSidebar
      owner={owner}
      open
      variant="docked"
      tabs={[]}
      activeTabId={null}
      onSelectTab={vi.fn()}
      onCloseTab={vi.fn()}
      onNewTab={vi.fn()}
      searchQuery=""
      onSearchQueryChange={vi.fn()}
      onOpenServerChat={onOpenServerChat}
      onOpenLocalHistory={onOpenLocalHistory}
      {...props}
    />
  )
  return { ...view, onOpenServerChat, onOpenLocalHistory }
}

const recentSection = () => screen.getByRole("region", { name: "Recent chats" })
const recentTitles = () =>
  within(recentSection())
    .getAllByRole("button")
    .map((button) => button.textContent)

beforeEach(() => {
  io.listeners.clear();
  io.readProfile.mockReset().mockResolvedValue({ history_profile_id: 'profile-alice' });
  io.serverHistory
    .mockReset()
    .mockImplementation((_query: string, options?: { mode?: string }) => ({
      data:
        options?.mode === "overview"
          ? [serverChat("s-new", "Server newest", 5_000), serverChat("s-old", "Server oldest", 1_000)]
          : [],
      isSuccess: true,
      isPending: false,
      isFetching: false,
      isError: false,
      fetchStatus: 'idle',
      refetch: vi.fn(),
    }));
  io.recentLocal
    .mockReset()
    .mockResolvedValue([
      localHistory('l-mid', 'Local middle', 3_000),
      localHistory('l-old', 'Local oldest', 500),
    ]);
  io.search.mockReset().mockResolvedValue([]);
});

describe('side-panel recent chats (XS-06)', () => {
  it('lists recent server and local chats, newest first, without a search', async () => {
    renderSidebar();

    await waitFor(() =>
      expect(recentTitles()).toEqual([
        "Server newestServer",
        "Local middleLocal",
        "Server oldestServer",
        "Local oldestLocal"
      ])
    )
    expect(io.serverHistory).toHaveBeenCalledWith(
      "",
      expect.objectContaining({ mode: "overview", page: 1, enabled: true })
    )
  })

  it("opens a recent server chat through the panel's server-chat opener", async () => {
    const { onOpenServerChat } = renderSidebar()

    fireEvent.click(await screen.findByRole("button", { name: /Server newest/ }))

    expect(onOpenServerChat).toHaveBeenCalledWith(expect.objectContaining({ id: "s-new", title: "Server newest" }))
  })

  it("opens a recent local chat through the panel's local-history opener", async () => {
    const { onOpenLocalHistory } = renderSidebar()

    fireEvent.click(await screen.findByRole("button", { name: /Local middle/ }))

    expect(onOpenLocalHistory).toHaveBeenCalledWith("l-mid")
  })

  it("leaves out chats that already have a tab and local copies of listed server chats", async () => {
    io.recentLocal.mockResolvedValue([
      localHistory("l-mid", "Local middle", 3_000),
      localHistory("l-open", "Local already open", 4_000),
      localHistory("l-mirror", "Server newest copy", 4_500, { server_chat_id: "s-new" })
    ])
    renderSidebar({
      tabs: [openTab("t1", { historyId: "l-open" }), openTab("t2", { serverChatId: "s-old" })]
    })

    await waitFor(() =>
      expect(recentTitles()).toEqual(["Server newestServer", "Local middleLocal"])
    )
  })

  it("shows only chats that belong to the signed-in account", async () => {
    io.recentLocal.mockResolvedValue([
      localHistory("l-mid", "Local middle", 3_000),
      localHistory("l-bob", "Bob's chat", 4_000, { server_scope_key: "bob" })
    ])
    renderSidebar()

    await waitFor(() => expect(recentTitles()).toContain("Local middleLocal"))
    expect(recentTitles().join(" ")).not.toContain("Bob's chat")
  })

  it("hides recents while searching", async () => {
    renderSidebar({ searchQuery: "planning" })

    await waitFor(() => expect(io.search).toHaveBeenCalled())
    expect(screen.queryByRole("region", { name: "Recent chats" })).not.toBeInTheDocument()
  })

  it("does not load recents until the account is verified", async () => {
    renderSidebar({ owner: { ...owner, isCurrent: () => false } })

    expect(screen.queryByRole("region", { name: "Recent chats" })).not.toBeInTheDocument()
    expect(io.recentLocal).not.toHaveBeenCalled()
    expect(io.serverHistory).not.toHaveBeenCalledWith("", expect.objectContaining({ mode: "overview", enabled: true }))
  })
})


it("refreshes recent owned copies after mutation without changing tabs or query", async () => {
  io.recentLocal
    .mockReset()
    .mockResolvedValue([localHistory("deleted", "Deleted device copy", 6_000)])
  const view = renderSidebar()
  await screen.findByText("Deleted device copy")
  io.recentLocal.mockResolvedValue([])
  act(() => io.listeners.forEach((listener) => listener(false)))
  await waitFor(() =>
    expect(screen.queryByText("Deleted device copy")).toBeNull()
  )
  view.unmount()
  expect(io.listeners.size).toBe(0)
})

it("retires a held recent read when a newer mutation read is pending", async () => {
  let release!: (rows: HistoryInfo[]) => void
  let releaseNext!: (rows: HistoryInfo[]) => void
  const held = new Promise<HistoryInfo[]>((resolve) => {
    release = resolve
  })
  const next = new Promise<HistoryInfo[]>((resolve) => {
    releaseNext = resolve
  })
  io.recentLocal.mockReset().mockReturnValueOnce(held).mockReturnValue(next)
  renderSidebar()
  await waitFor(() => expect(io.recentLocal).toHaveBeenCalledTimes(1))
  await act(async () => {
    io.listeners.forEach((listener) => listener(false))
    release([localHistory("deleted", "Late deleted copy", 6_000)])
  })
  await waitFor(() => expect(io.recentLocal).toHaveBeenCalledTimes(2))
  expect(screen.queryByText("Late deleted copy")).toBeNull()
  await act(async () => {
    releaseNext([])
  })
  expect(screen.queryByText("Late deleted copy")).toBeNull()
})

it('keeps an owned server mirror origin when its conversation is outside the overview page', async () => {
  io.recentLocal.mockResolvedValue([
    localHistory('mirror', 'Missing from overview', 6_000, {
      server_chat_id: 's-unlisted',
      message_source: 'server',
      server_scope_type: 'global',
      local_owner_key: 'local-history-v1:profile-alice',
    }),
    localHistory('local', 'Actual local', 5_500, {
      local_owner_key: 'local-history-v1:profile-alice',
    }),
  ]);
  const { onOpenLocalHistory, onOpenServerChat } = renderSidebar();
  const mirror = await screen.findByRole('button', {
    name: /^Missing from overview\s*Server copy$/,
  });
  fireEvent.click(mirror);
  expect(onOpenLocalHistory).toHaveBeenCalledWith('mirror');
  expect(onOpenServerChat).not.toHaveBeenCalled();
  expect(recentTitles()).toContain('Actual localLocal');
});

it.each([
  { server_scope_key: 'bob' },
  { server_scope_key: undefined },
  { local_owner_key: 'local-history-v1:foreign-profile' },
  { server_scope_type: 'workspace' as const, server_workspace_id: 'other-space' },
  { server_scope_type: 'global' as const, server_workspace_id: 'malformed-space' },
  { server_chat_id: '' },
])('does not offer an unqualified or unsupported-namespace mirror (%j)', async (extra) => {
  io.recentLocal.mockResolvedValue([
    localHistory('invalid', 'Unqualified server copy', 6_000, {
      server_chat_id: 's-unlisted',
      message_source: 'server',
      ...extra,
    }),
  ]);
  renderSidebar();
  await waitFor(() => expect(io.recentLocal).toHaveBeenCalled());
  await waitFor(() => expect(recentTitles()).toContain('Server newestServer'));
  expect(recentTitles().join(' ')).not.toContain('Unqualified server copy');
});

it('retires a held owned mirror read when the account owner is replaced', async () => {
  let release!: (rows: HistoryInfo[]) => void;
  const held = new Promise<HistoryInfo[]>((resolve) => {
    release = resolve;
  });
  let current = true;
  io.recentLocal.mockReturnValueOnce(held);
  const view = renderSidebar({ owner: { ...owner, isCurrent: () => current } });
  await waitFor(() => expect(io.recentLocal).toHaveBeenCalled());
  current = false;
  await act(async () => {
    release([
      localHistory('mirror', 'Retired mirror', 6_000, {
        server_chat_id: 's-retired',
        message_source: 'server',
      }),
    ]);
  });
  expect(screen.queryByText('Retired mirror')).toBeNull();
  view.unmount();
});

it('retires a held profile read before its histories can be offered after a profile mutation', async () => {
  let release!: (settings: { history_profile_id: string }) => void;
  const held = new Promise<{ history_profile_id: string }>((resolve) => {
    release = resolve;
  });
  io.readProfile.mockReturnValueOnce(held).mockResolvedValue({ history_profile_id: 'profile-bob' });
  io.recentLocal.mockResolvedValue([
    localHistory('old-profile', 'Former profile', 6_000, {
      local_owner_key: 'local-history-v1:profile-alice',
    }),
  ]);
  const view = renderSidebar();
  await waitFor(() => expect(io.readProfile).toHaveBeenCalledOnce());
  act(() => io.listeners.forEach((listener) => listener(true)));
  await act(async () => {
    release({ history_profile_id: 'profile-alice' });
  });
  await waitFor(() => expect(io.readProfile).toHaveBeenCalledTimes(2));
  expect(screen.queryByText('Former profile')).toBeNull();
  view.unmount();
});

it('qualifies the profile and global namespace before the bounded recent read', async () => {
  const crowded = Array.from({ length: 20 }, (_, index) =>
    localHistory(`workspace-${index}`, `Workspace ${index}`, 7_000 - index, {
      server_chat_id: `s-workspace-${index}`,
      message_source: 'server',
      server_scope_type: 'workspace',
      server_workspace_id: 'other-space',
    })
  );
  const eligible = localHistory('eligible', 'Eligible global mirror', 6_000, {
    server_chat_id: 's-eligible',
    message_source: 'server',
    server_scope_type: 'global',
    local_owner_key: 'local-history-v1:profile-alice',
  });
  io.recentLocal.mockImplementation(
    async (limit: number, filter: (history: HistoryInfo) => boolean) =>
      [...crowded, eligible].filter(filter).slice(0, limit)
  );
  renderSidebar();
  await screen.findByRole('button', { name: /^Eligible global mirror\s*Server copy$/ });
  expect(recentTitles().join(' ')).not.toContain('Workspace');
});
