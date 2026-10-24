import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryInfo, Message } from "../types";

// The production query and qualification run; only IndexedDB is replaced here.
// Browser verification separately covers real transaction/liveQuery behavior.
const state = vi.hoisted(() => ({
  rows: [] as HistoryInfo[],
  messages: [] as Message[],
  profile: "P" as string | null,
  fail: false,
  afterRead: null as (() => void) | null,
  writes: vi.fn(),
}));
function collection<T>(input: () => T[]) {
  let select = input;
  const api = {
    reverse() {
      const previous = select;
      select = () => [...previous()].reverse();
      return api;
    },
    filter(test: (row: T) => boolean) {
      const previous = select;
      select = () => previous().filter(test);
      return api;
    },
    offset(n: number) {
      const previous = select;
      select = () => previous().slice(n);
      return api;
    },
    limit(n: number) {
      const previous = select;
      select = () => previous().slice(0, n);
      return api;
    },
    async toArray() {
      if (state.fail) throw new Error("storage unavailable");
      const result = structuredClone(select());
      state.afterRead?.();
      return result;
    },
    async each(callback: (row: T) => void) {
      if (state.fail) throw new Error("storage unavailable");
      const rows = structuredClone(select());
      state.afterRead?.();
      rows.forEach(callback);
    },
    async count() {
      return select().length;
    },
  };
  return api;
}
function table<T extends { id: string }>(rows: () => T[]) {
  return {
    get: async (id: string) => rows().find((row) => row.id === id),
    count: async () => rows().length,
    toArray: () => collection(rows).toArray(),
    orderBy: (field: keyof T) =>
      collection(() =>
        [...rows()].sort((a, b) => Number(a[field]) - Number(b[field])),
      ),
    filter: (test: (row: T) => boolean) => collection(rows).filter(test),
    where: (field: keyof T) => ({
      equals: (value: unknown) =>
        collection(rows).filter((row) => row[field] === value),
      anyOf: (values: unknown[]) =>
        collection(rows).filter((row) => values.includes(row[field])),
      startsWithIgnoreCase: (query: string) => ({
        or: () => ({
          anyOfIgnoreCase: (words: string[]) =>
            collection(rows).filter(
              (row) =>
                String(row[field]).toLowerCase().startsWith(query) ||
                words.includes(String(row[field]).toLowerCase()),
            ),
        }),
      }),
    }),
    put: state.writes,
    add: state.writes,
    update: state.writes,
    delete: state.writes,
  };
}
// Token side-table for content search (WP-23): candidates are derived from
// the seeded messages with the production tokenizer, mirroring a database
// whose write paths (or v17 backfill) kept the index fresh. Declared as a
// function so the hoisted vi.mock factory can reference it.
function searchTokensTable(tokenize: (text: string) => string[]) {
  return {
    where: (field: string) => ({
      anyOf: (tokens: string[]) => ({
        primaryKeys: async () => {
          if (field !== "token") throw new Error(`searchTokens.where(${field}) unsupported`);
          const wanted = new Set(tokens);
          const seen = new Set<string>();
          const keys: Array<[string, string]> = [];
          for (const message of state.messages) {
            for (const token of tokenize(message.content)) {
              if (!wanted.has(token)) continue;
              const pair = `${token}|${message.history_id}`;
              if (seen.has(pair)) continue;
              seen.add(pair);
              keys.push([token, message.history_id]);
            }
          }
          return keys;
        }
      })
    })
  };
}
vi.mock("../schema", async () => {
  const { tokenizeSearchText } = await import("../search-tokenizer");
  return {
  db: {
    name: "PageAssistDatabase",
    chatHistories: table(() => state.rows),
    messages: table(() => state.messages),
    searchTokens: searchTokensTable(tokenizeSearchText),
    userSettings: {
      get: async () =>
        state.profile
          ? { id: "main", history_profile_id: state.profile }
          : undefined,
      put: state.writes,
    },
    modelNickname: { toArray: async () => [] },
    transaction: async (
      _mode: string,
      _tables: unknown[],
      fn: () => Promise<unknown>,
    ) => fn(),
  },
  };
});
import Dexie from "dexie";
import { PageAssistDatabase, watchOnDeviceHistoryChanges } from "../chat";
const database = new PageAssistDatabase();
const row = (id: string, extra: Partial<HistoryInfo> = {}): HistoryInfo => ({
  id,
  title: id,
  createdAt: (Number(id.replace(/\D/g, "")) || 0) + 1,
  is_rag: false,
  local_owner_key: "local-history-v1:P",
  ...extra,
});
const scope = (extra = {}) => ({
  profileId: "P",
  serverScopeKey: "A",
  isCurrent: () => true,
  ...extra,
});
const message = (id: string, history_id: string, content: string): Message => ({
  id,
  history_id,
  content,
  createdAt: 1,
  role: "user",
  name: "You",
});
beforeEach(() => {
  state.rows = [];
  state.messages = [];
  state.profile = "P";
  state.fail = false;
  state.afterRead = null;
  state.writes.mockClear();
});

describe("qualified on-device history reads", () => {
  it("filters foreign, mirrored, unowned and Temporary records before counts and pages", async () => {
    state.rows = Array.from({ length: 35 }, (_, i) => row(`owned-${i}`));
    state.rows.push(
      row("mirror", { server_chat_id: "server" }),
      row("source", { message_source: "server" }),
      row("temp"),
      row("foreign", {
        local_owner_key: "local-history-v1:Q",
        server_scope_key: "A",
      }),
      row("unowned", { local_owner_key: undefined }),
      row("realm-B", { server_scope_key: "B" }),
      row("realm-A", { local_owner_key: undefined, server_scope_key: "A" }),
    );
    const first = await database.getChatHistoriesPaginated(
      1,
      undefined,
      scope(),
    );
    const second = await database.getChatHistoriesPaginated(
      2,
      undefined,
      scope(),
    );
    expect([
      first.totalCount,
      first.histories.length,
      first.hasMore,
      second.histories.length,
      second.hasMore,
    ]).toEqual([36, 30, true, 6, false]);
    expect(
      [...first.histories, ...second.histories].every(
        (item) =>
          ![
            "mirror",
            "source",
            "temp",
            "foreign",
            "unowned",
            "realm-B",
          ].includes(item.id),
      ),
    ).toBe(true);
    expect(state.writes).not.toHaveBeenCalled();
  });
  it("does not offer another page for exactly thirty eligible chats", async () => {
    state.rows = Array.from({ length: 30 }, (_, i) => row(`owned-${i}`));
    expect(
      (await database.getChatHistoriesPaginated(1, undefined, scope())).hasMore,
    ).toBe(false);
  });
  it("searches title and message content beyond the first page then qualifies results", async () => {
    state.rows = Array.from({ length: 70 }, (_, i) => row(`owned-${i}`));
    state.rows[0].title = "Needle in a title";
    state.rows.push(
      row("foreign", {
        title: "Needle foreign",
        local_owner_key: "local-history-v1:Q",
      }),
    );
    state.messages = [
      message("content", "owned-1", "A needle in a message"),
      message("secret", "foreign", "Needle foreign body"),
    ];
    const result = await database.getChatHistoriesPaginated(
      1,
      "  needle  ",
      scope(),
    );
    expect(result.histories.map((item) => item.id)).toEqual([
      "owned-1",
      "owned-0",
    ]);
    expect(result.totalCount).toBe(2);
  });
  it("allows profile-only forks without an inference account but hides realm-bound rows", async () => {
    state.rows = [row("profile"), row("realm", { server_scope_key: "A" })];
    expect(
      (
        await database.getChatHistoriesPaginated(
          1,
          undefined,
          scope({ serverScopeKey: null }),
        )
      ).histories.map((item) => item.id),
    ).toEqual(["profile"]);
  });
  it("reads matching scope-only chats without initializing or adopting a profile", async () => {
    state.profile = null;
    state.rows = [
      row("realm", { local_owner_key: undefined, server_scope_key: "A" }),
      row("unowned", { local_owner_key: undefined }),
    ];
    const result = await database.getChatHistoriesPaginated(
      1,
      undefined,
      scope({ profileId: null }),
    );
    expect(result.histories.map((item) => item.id)).toEqual(["realm"]);
    expect(state.writes).not.toHaveBeenCalled();
  });
  it("rejects a changed profile before returning data", async () => {
    state.rows = [row("secret")];
    state.profile = "Q";
    await expect(
      database.getChatHistoriesPaginated(1, undefined, scope()),
    ).rejects.toThrow();
  });
  it("rejects a lost caller lease after a held read", async () => {
    state.rows = [row("secret")];
    let current = true;
    state.afterRead = () => {
      current = false;
    };
    await expect(
      database.getChatHistoriesPaginated(
        1,
        undefined,
        scope({ isCurrent: () => current }),
      ),
    ).rejects.toThrow();
  });
  it("propagates storage failure instead of returning an empty page", async () => {
    state.fail = true;
    await expect(
      database.getChatHistoriesPaginated(1, undefined, scope()),
    ).rejects.toThrow("storage unavailable");
  });
});

it("reads all stored preview rows without adopting a path or writing profile/bookmarks", async () => {
  state.rows = [row("owned")];
  state.messages = [
    message("b", "owned", "Alternative B"),
    message("a", "owned", "Alternative A"),
  ];
  const preview = await database.getOnDeviceChatPreview("owned", scope());
  expect(preview?.messages.map((item) => item.id)).toEqual(["a", "b"]);
  expect(state.writes).not.toHaveBeenCalled();
});
it.each(["foreign", "mirror", "unowned", "temp"])(
  "refuses a preview of %s",
  async (id) => {
    state.rows = [
      row("foreign", { local_owner_key: "local-history-v1:Q" }),
      row("mirror", { server_chat_id: "server" }),
      row("unowned", { local_owner_key: undefined }),
      row("temp"),
    ];
    state.messages = [message("secret", id, "Hidden")];
    expect(await database.getOnDeviceChatPreview(id, scope())).toBeNull();
    expect(state.writes).not.toHaveBeenCalled();
  },
);
it("qualifies metadata excerpts again rather than trusting supplied IDs", async () => {
  state.rows = [
    row("owned"),
    row("foreign", { local_owner_key: "local-history-v1:Q" }),
  ];
  state.messages = [
    message("a", "owned", "Available"),
    message("b", "foreign", "Secret"),
  ];
  const metadata = await database.getHistoriesWithMetadata(
    ["owned", "foreign"],
    scope(),
  );
  expect([...metadata.keys()]).toEqual(["owned"]);
});

it("observes only relevant DB changes, including unchanged-count edits, and unsubscribes", () => {
  const changed = vi.fn();
  const stop = watchOnDeviceHistoryChanges(changed);
  const fire = (part: string) =>
    Dexie.on.storagemutated.fire({ [part]: {} as never });
  fire("idb://OtherDatabase/chatHistories/");
  fire("idb://PageAssistDatabase/prompts/");
  expect(changed).not.toHaveBeenCalled();
  fire("idb://PageAssistDatabase/chatHistories/");
  fire("idb://PageAssistDatabase/messages/");
  fire("idb://PageAssistDatabase/userSettings/");
  expect(changed.mock.calls).toEqual([[false], [false], [true]]);
  stop();
  fire("idb://PageAssistDatabase/chatHistories/");
  expect(changed).toHaveBeenCalledTimes(3);
});


describe("qualified cached server discovery", () => {
  const cached = (id: string, chatId = id, extra = {}) => row(id, {
    message_source: "server", server_chat_id: chatId, server_scope_key: "A",
    server_scope_type: "global", ...extra
  })
  const policy = () => ({ includeCachedServerChats: true })
  it("includes only owned account cached copies under an explicit read policy", async () => {
    state.rows = [row("local"), cached("copy"), cached("foreign-account", "f", {server_scope_key: "B"}),
      cached("foreign-profile", "p", {local_owner_key:"local-history-v1:Q"}),
      cached("unbound", "u", {server_scope_key:undefined}), row("temp", {server_chat_id:"t", server_scope_key:"A"})]
    const result = await database.getChatHistoriesPaginated(1, undefined, scope(), policy())
    expect(result.histories.map(item => item.id).sort()).toEqual(["copy", "local"])
    expect(result.totalCount).toBe(2)
    expect(state.writes).not.toHaveBeenCalled()
  })
  it("deduplicates only visible current global server identity before counts and pages", async () => {
    state.rows = [cached("matched", "S"), cached("other-id", "T"), cached("unknown-scope", "S", {server_scope_type: undefined}),
      cached("workspace", "S", {server_scope_type: "workspace", server_workspace_id:"W"}),
      row("same-title", {title:"matched"})]
    const visibleServerChats = {serverScopeKey:"A", query:"", ids:new Set(["S"]), isCurrent:()=>true}
    const result = await database.getChatHistoriesPaginated(1, undefined, scope(), { ...policy(), visibleServerChats })
    expect(result.histories.map(item => item.id).sort()).toEqual(["other-id", "same-title", "unknown-scope", "workspace"])
    expect(result.totalCount).toBe(4)
  })
  it.each(["stale", "other account", "other query", "other page"])("keeps copies reachable for %s server results", async reason => {
    state.rows = [cached("copy", "S", {title:"needle"})]
    const visibleServerChats = {serverScopeKey:reason === "other account" ? "B" : "A", query:reason === "other query" ? "else" : "needle",
      ids:new Set(reason === "other page" ? ["T"] : ["S"]), isCurrent:()=>reason !== "stale"}
    const result = await database.getChatHistoriesPaginated(1, "needle", scope(), {...policy(), visibleServerChats})
    expect(result.histories.map(item => item.id)).toEqual(["copy"])
  })
  it("previews cached stored rows without adopting a local path or writing bookmarks", async () => {
    state.rows = [cached("copy", "S")]; state.messages = [message("q", "copy", "Stored cached question")]
    const result = await database.getOnDeviceChatPreview("copy", scope(), policy())
    expect(result?.messages.map(item => item.content)).toEqual(["Stored cached question"])
    expect(await database.getOnDeviceChatPreview("copy", scope())).toBeNull()
    expect(state.writes).not.toHaveBeenCalled()
  })
  it("qualifies and pages 2000 local/cached histories and searches their content", async () => {
    state.rows = Array.from({length:2000}, (_, i) => i < 1100 ? row(`local-${i}`) : cached(`copy-${i}`, `S-${i}`))
    state.rows.push(cached("foreign", "F", {server_scope_key:"B"}))
    state.messages = [message("needle", "copy-1999", "Distinctive cached body")]
    const ids = new Set<string>()
    for (let page = 1; page <= 67; page++) {
      const result = await database.getChatHistoriesPaginated(page, undefined, scope(), policy())
      result.histories.forEach(item => ids.add(item.id))
      expect(result.hasMore).toBe(page < 67)
      expect(result.totalCount).toBe(2000)
    }
    expect(ids.size).toBe(2000)
    expect((await database.getChatHistoriesPaginated(1, "distinctive", scope(), policy())).histories.map(item => item.id)).toEqual(["copy-1999"])
  })
})
