// @vitest-environment jsdom
import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App, ConfigProvider, message } from "antd";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { KanbanPlayground } from "../index";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";

const transport = vi.hoisted(() => ({
  canonical: vi.fn(),
  getConfig: vi.fn(),
  get: vi.fn(),
  fetch: vi.fn(),
  catalog: vi.fn(),
}));
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: transport.canonical,
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: transport.getConfig },
}));
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }));
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: transport.get,
    set: vi.fn(),
    remove: vi.fn(),
    watch: vi.fn(),
    unwatch: vi.fn(),
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value,
  },
}));
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false,
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }),
}));
vi.mock("@/components/Common/ModelSelect", () => ({
  ModelSelect: () => <span>Fixture model picker</span>,
}));
vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: "llamacpp:fixture/Qwen:Q8_0",
    selectedModelIsLoading: false,
  }),
}));
vi.mock("@/hooks/useChatModelCatalog", () => ({
  useChatModelCatalog: transport.catalog,
}));
vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (select: (value: unknown) => unknown) =>
    select({
      apiProvider: "llamacpp",
      temperature: 0.2,
      topP: 0.9,
      numPredict: 128,
    }),
}));
vi.mock("@dnd-kit/react", () => ({
  DragDropProvider: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  DragOverlay: () => null,
}));
vi.mock("@dnd-kit/react/sortable", () => ({
  useSortable: () => ({ ref: vi.fn(), handleRef: vi.fn(), isDragging: false }),
}));

const generated = {
  board: { name: "Rowan Tasks", description: "Controlled source" },
  lists: [
    {
      name: "To Do",
      cards: [
        { title: "Calibrate Selene", description: "Telescope named Selene" },
        { title: "Prepare Friday tours", description: "Tours begin at 18:00" },
      ],
    },
    {
      name: "Done",
      cards: [
        { title: "Record 2019 opening", description: "Rowan Observatory" },
      ],
    },
  ],
};
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const completion = (value: unknown = generated) =>
  json({
    choices: [
      { message: { content: "```json\n" + JSON.stringify(value) + "\n```" } },
    ],
  });
const config = (
  owner = 2,
  serverUrl = "https://kanban.example.test",
): TldwConfig => ({
  serverUrl,
  authMode: "multi-user",
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(owner) }))}.signature`,
});
let pendingResponses: { resolve: (value: Response) => void }[] = [];
const deferred = () => {
  let resolve!: (value: Response) => void;
  const promise = new Promise<Response>((done) => {
    resolve = done;
  });
  pendingResponses.push({ resolve });
  return { promise, resolve };
};
let current: TldwConfig;
let client: QueryClient;
type SaveDraft = typeof generated & { client_id: string; format: string };
type SavedRequest = {
  path: string;
  init: RequestInit;
  body: { data: SaveDraft };
};
let records: {
  path: string;
  init: RequestInit;
  body: Record<string, unknown>;
}[];
let saved: ReturnType<typeof completeBoard> | null;
let chat: (init: RequestInit) => Response | Promise<Response>;
let save: (body: { data: SaveDraft }) => Response | Promise<Response>;
const completeBoard = (body: SaveDraft) => ({
  ...body.board,
  id: 1,
  uuid: "saved-board",
  user_id: "2",
  client_id: body.client_id,
  archived: false,
  deleted: false,
  version: 1,
  created_at: "2026-10-27T00:00:00Z",
  updated_at: "2026-10-27T00:00:00Z",
  labels: [],
  total_cards: 3,
  lists: body.lists.map((row, index: number) => ({
    ...row,
    id: index + 1,
    uuid: `list-${index}`,
    client_id: `list-client-${index}`,
    board_id: 1,
    position: index,
    archived: false,
    version: 1,
    cards: row.cards.map((card, order: number) => ({
      ...card,
      id: index * 10 + order + 1,
      uuid: `card-${index}-${order}`,
      client_id: `card-client-${index}-${order}`,
      list_id: index + 1,
      board_id: 1,
      position: order,
      version: 1,
      archived: false,
      deleted: false,
      labels: [],
      created_at: "2026-10-27T00:00:00Z",
      updated_at: "2026-10-27T00:00:00Z",
    })),
  })),
});
const tree = () => (
  <QueryClientProvider client={client}>
    <ConfigProvider theme={{ token: { motion: false } }}>
      <App>
        <KanbanPlayground />
      </App>
    </ConfigProvider>
  </QueryClientProvider>
);
async function open() {
  const view = render(tree());
  fireEvent.click(
    await screen.findByRole("button", { name: "Generate board", exact: true }),
  );
  // AntD gives its selector and dialog the same test-mode ID; check the dialog and title together.
  const dialog = await screen.findByRole("dialog");
  await waitFor(() => expect(dialog).toBeVisible());
  expect(
    within(dialog).getByText("Generate board", { exact: true }),
  ).toBeVisible();
  fireEvent.change(
    within(dialog).getByRole("textbox", { name: "Board instructions" }),
    { target: { value: "Create two lists and three Rowan tasks." } },
  );
  return { view, dialog };
}
async function generate(dialog: HTMLElement) {
  fireEvent.click(
    within(dialog).getByRole("button", { name: "Generate draft", exact: true }),
  );
  await waitFor(() =>
    expect(
      within(dialog).getByRole("button", {
        name: "Generate draft",
        exact: true,
      }),
    ).toBeEnabled(),
  );
  return await within(dialog).findByRole("region", {
    name: "Generated board preview",
  });
}
const requests = (path: string) =>
  records.filter((row) => row.path === path && row.init.method === "POST");
const chats = () => requests("/api/v1/chat/completions");
const saves = () => requests("/api/v1/kanban/boards/import") as SavedRequest[];

beforeEach(() => {
  vi.resetAllMocks();
  current = config();
  records = [];
  saved = null;
  pendingResponses = [];
  client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  transport.canonical.mockImplementation(() => ({
    config: current,
    authorityLoading: false,
  }));
  transport.catalog.mockReturnValue({
    data: [{ model: "fixture/Qwen:Q8_0", provider: "llamacpp" }],
    isSuccess: true,
    isFetching: false,
  });
  transport.getConfig.mockImplementation(async () => current);
  transport.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? current : null,
  );
  chat = () => completion();
  save = (body) => {
    saved = completeBoard(body.data);
    return json(
      {
        board: saved,
        import_stats: { board_id: 1, lists_imported: 2, cards_imported: 3 },
      },
      201,
    );
  };
  transport.fetch.mockImplementation(
    async (input: string, init: RequestInit = {}) => {
      const path = new URL(String(input)).pathname;
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      records.push({ path, init, body });
      if (path === "/api/v1/kanban/boards")
        return json({
          boards: saved ? [saved] : [],
          pagination: { total: saved ? 1 : 0 },
        });
      if (path === "/api/v1/kanban/boards/1") return json(saved);
      if (path.endsWith("/labels")) return json([]);
      if (path === "/api/v1/chat/completions") return await chat(init);
      if (path === "/api/v1/kanban/boards/import") return await save(body);
      throw new Error(`Unexpected fixture request ${init.method} ${path}`);
    },
  );
  vi.stubGlobal("fetch", transport.fetch);
});
afterEach(async () => {
  cleanup();
  await act(async () => {
    for (const pending of pendingResponses) pending.resolve(completion());
  });
  client.clear();
  message.destroy();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("generated Kanban drafts", () => {
  it.each([
    ["empty", []],
    ["missing selection", [{ model: "another/model", provider: "llamacpp" }]],
    [
      "unavailable",
      [{ model: "fixture/Qwen:Q8_0", provider: "llamacpp", available: false }],
    ],
  ])("blocks a stale model when the catalog is %s", async (_label, models) => {
    transport.catalog.mockReturnValue({
      data: models,
      isSuccess: true,
      isFetching: false,
    });
    const { dialog } = await open();
    const button = within(dialog).getByRole("button", {
      name: "Generate draft",
      exact: true,
    });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(chats()).toHaveLength(0);
    expect(saves()).toHaveLength(0);
  });

  it("previews every list/card and saves only after an explicit action", async () => {
    const { dialog } = await open();
    const preview = await generate(dialog);
    for (const text of [
      "Rowan Tasks",
      "To Do",
      "Done",
      "Calibrate Selene",
      "Prepare Friday tours",
      "Record 2019 opening",
    ])
      expect(within(preview).getByText(text, { exact: true })).toBeVisible();
    expect(saves()).toHaveLength(0);
    expect(chats()[0].body).toMatchObject({
      model: "fixture/Qwen:Q8_0",
      api_provider: "llamacpp",
      max_tokens: 128,
      temperature: 0.2,
      top_p: 0.9,
      stream: false,
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Save board", exact: true }),
    );
    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    expect(saves()).toHaveLength(1);
    expect(saves()[0].body.data).toMatchObject({
      format: "tldw_kanban_draft_v1",
      ...generated,
    });
    expect(saves()[0].body.data.client_id).toEqual(expect.any(String));
  });

  it("stops the request and rejects a late result without saving", async () => {
    const pending = deferred();
    chat = () => pending.promise;
    const { dialog } = await open();
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: "Generate draft",
        exact: true,
      }),
    );
    await waitFor(() => expect(chats()).toHaveLength(1));
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Stop", exact: true }),
    );
    expect(chats()[0].init.signal?.aborted).toBe(true);
    await act(async () => {
      pending.resolve(completion());
    });
    expect(
      within(dialog).queryByRole("region", { name: "Generated board preview" }),
    ).not.toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: "Save board", exact: true }),
    ).toBeDisabled();
    expect(saves()).toHaveLength(0);
  });

  it("rejects model-supplied database identities and never enables Save", async () => {
    const invalid = structuredClone(generated);
    (invalid.lists[0].cards[0] as Record<string, unknown>).id = "foreign-card";
    chat = () => completion(invalid);
    const { dialog } = await open();
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: "Generate draft",
        exact: true,
      }),
    );
    await waitFor(() =>
      expect(within(dialog).getByRole("alert")).toHaveTextContent(
        /valid|invalid/i,
      ),
    );
    expect(
      within(dialog).queryByRole("region", { name: "Generated board preview" }),
    ).not.toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: "Save board", exact: true }),
    ).toBeDisabled();
    expect(saves()).toHaveLength(0);
  });

  it("preserves the prior preview on provider failure and permits an explicit retry", async () => {
    const { dialog } = await open();
    await generate(dialog);
    chat = () => json({ detail: "provider_unavailable" }, 503);
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: "Generate draft",
        exact: true,
      }),
    );
    await waitFor(() =>
      expect(within(dialog).getByRole("alert")).toHaveTextContent(
        /provider|unavailable/i,
      ),
    );
    expect(
      within(dialog).getByRole("region", { name: "Generated board preview" }),
    ).toHaveTextContent("Calibrate Selene");
    expect(saves()).toHaveLength(0);
    chat = () => completion();
    await generate(dialog);
    expect(chats()).toHaveLength(3);
  });

  it("keeps a pending Save visible and suppresses duplicate submission", async () => {
    const pending = deferred();
    save = () => pending.promise;
    const { dialog } = await open();
    await generate(dialog);
    const button = within(dialog).getByRole("button", {
      name: "Save board",
      exact: true,
    });
    fireEvent.click(button);
    fireEvent.click(button);
    await waitFor(() => expect(saves()).toHaveLength(1));
    expect(dialog).toBeVisible();
    expect(
      within(dialog).getByRole("button", { name: "Cancel", exact: true }),
    ).toBeDisabled();
    await act(async () => {
      saved = completeBoard(saves()[0].body.data);
      pending.resolve(
        json({ board: saved, import_stats: { board_id: 1 } }, 201),
      );
    });
    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    expect(saves()).toHaveLength(1);
  });

  it("retains a failed Save and retries with the same draft identity", async () => {
    let attempts = 0;
    save = (body) => {
      attempts += 1;
      if (attempts === 1)
        return json({ detail: "Storage temporarily unavailable" }, 500);
      saved = completeBoard(body.data);
      return json({ board: saved, import_stats: { board_id: 1 } }, 201);
    };
    const { dialog } = await open();
    await generate(dialog);
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Save board", exact: true }),
    );
    await waitFor(() =>
      expect(within(dialog).getByRole("alert")).toHaveTextContent(
        /Try Save board again/,
      ),
    );
    expect(
      within(dialog).getByRole("region", { name: "Generated board preview" }),
    ).toHaveTextContent("Calibrate Selene");
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Save board", exact: true }),
    );
    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    expect(chats()).toHaveLength(1);
    expect(saves()).toHaveLength(2);
    expect(saves()[1].body.data).toEqual(saves()[0].body.data);
  });

  it.each(["account", "server"])(
    "retires a held draft after %s changes",
    async (change) => {
      const pending = deferred();
      chat = () => pending.promise;
      const { view, dialog } = await open();
      fireEvent.click(
        within(dialog).getByRole("button", {
          name: "Generate draft",
          exact: true,
        }),
      );
      await waitFor(() => expect(chats()).toHaveLength(1));
      current =
        change === "account"
          ? config(3)
          : config(3, "https://bob.example.test");
      view.rerender(tree());
      await waitFor(() => expect(dialog).not.toBeInTheDocument());
      expect(chats()[0].init.signal?.aborted).toBe(true);
      await act(async () => {
        pending.resolve(completion());
      });
      expect(screen.queryByText("Calibrate Selene")).not.toBeInTheDocument();
      expect(saves()).toHaveLength(0);
    },
  );
});
